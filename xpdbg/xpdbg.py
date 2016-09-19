#!/usr/bin/env python
"""
	XPDBG -- Runtime Analysis Toolkit for PHP
	by Romain Gaucher <r@rgaucher.info> - http://rgaucher.info

	Copyright (c) 2009-2012 Romain Gaucher <r@rgaucher.info>

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
"""
import sys, time, os, re, math, json
from twisted.conch.ssh import transport, userauth, connection, channel
from twisted.conch.ssh.common import NS
from twisted.internet import defer, protocol, reactor
from twisted.python import log
from getpass import getpass
import struct, subprocess, platform, logging

from pygraph.classes.digraph import digraph
from pygraph.mixins.labeling import labeling
from pygraph.algorithms.cycles import find_cycle
from pygraph.algorithms.accessibility import mutual_accessibility, connected_components
from pygraph.readwrite.markup import write as write_xml
from pygraph.readwrite.dot import write as write_dot

import sqlite3


from pygments import highlight
from pygments.lexers import PhpLexer
from pygments.formatters import HtmlFormatter
from pygments.styles import get_style_by_name

__isiterable = lambda obj: isinstance(obj, basestring) or getattr(obj, '__iter__', False)
__normalize_argmt = lambda x: ''.join(x.lower().split())
__normalize_paths = lambda x: [os.path.abspath(of) for of in x]

import logging
import logging.config
logging.config.fileConfig("log.conf")
log = logging.getLogger("xpdbg")

# DB storage of needles
unique_id_db = 'test-ids.db'

BLACK_LIST = ('FOOBAR',)
def has_blacklisted_function(rts):
	rts = rts.lower()
	for elmt in BLACK_LIST:
		if elmt in rts:
			return True
	return False

def create_db():
	ret = False
	if not os.path.isfile(unique_id_db):
		log.debug("create unique_id_db database")
		conn = sqlite3.connect(unique_id_db)
		conn.isolation_level = None
		c = conn.cursor()
		c.execute("""create table rtid (data text)""")
		conn.commit()
		c.close()
		conn.close()


def db_list_rtid():
	conn = sqlite3.connect(unique_id_db)
	conn.isolation_level = None
	c = conn.cursor()
	c.execute("""select data from rtid""")
	log.debug("fetch content from unique_id_db DB")
	ret = []
	for r in c:
		if r not in ret:
			ret.append(r[0])
	c.close()
	conn.close()
	return ret


def pygment_content(rts):
	return highlight(rts, PhpLexer(), HtmlFormatter(style='monokai', cssclass="source", full=True))


def syscall(bin, parameters):
	# system call, wait for process to finish and capture the stdout/stderr output
	syscmd = ' '.join([bin] + parameters)
	
	log.debug("execute command: %s" % syscmd)
	
	p = subprocess.Popen(syscmd, bufsize=65536, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	retcode = 0
	if 'Windows' != platform.system():
		retcode = p.wait()
	(child_stdout, child_stderr) = (p.stdout, p.stderr)
	return retcode, (child_stdout.read(), child_stderr.read())


USER, HOST, CMD = None, None, None
# security feature, pwd in source code
PASSWD = 'i am so stupid'

class Transport(transport.SSHClientTransport):
	def verifyHostKey(self, hostKey, fingerprint):
		log.debug("Connection to %s successful..." % (HOST))
		return defer.succeed(1)

	def connectionSecure(self):
		self.requestService(UserAuth(USER, Connection()))

class UserAuth(userauth.SSHUserAuthClient):
	def getPassword(self):
		return defer.succeed(PASSWD)

	def getPublicKey(self):
		return

class Connection(connection.SSHConnection):
	def serviceStarted(self):
		self.openChannel(Channel(2**16, 2**15, self))

class Channel(channel.SSHChannel):
	name = 'session'	# must use this exact string
	def openFailed(self, reason):
		print '"%s" failed: %s' % (CMD,reason)

	def channelOpen(self, data):
		print "Executing %s..." % CMD
		self.model = Modeler()
		self.exptime = time.time()
		d = self.conn.sendRequest(self, 'exec', NS(CMD), wantReply=1)

	def dataReceived(self, data):
		recs = data.strip().split('\n')
		for rec in recs:
			self.model.feedAccumulator(rec)
			
		if self.model.hasEnoughData():
			log.debug("Model has enough data to execute the loop...")
			self.model.run()

	def closed(self):
		try:
			self.loseConnection()
			self.model.export()
			reactor.stop()
		except:
			# Nasty return, who cares?
			return


# literate stack implementation
class EmptyStackException(Exception):
	pass

class Element:
	def __init__(self, value, next):
		self.value = value
		self.next = next

class Stack:
	def __init__(self):
		self.head = None

	def push(self, element):
		self.head = Element(element, self.head)

	def pop(self):
		if self.empty(): raise EmptyStackException
		result = self.head.value
		self.head = self.head.next
		return result

	def top(self):
		return self.head.value

	def empty(self):
		return self.head == None


def xml_entities(rts):
	return rts.replace('>', '&gt;').replace('<', '&lt;').replace('"', '&quot;')


TOKEN_INITIATOR = '>'

TOKEN_FCT_CALL = 1
TOKEN_ASSIGNMENT = 2
TOKEN_RETURN = 3

MAIN = '{main}()'
TOKENS = {
	'->' : TOKEN_FCT_CALL,    # function call, enter context
	'=>' : TOKEN_ASSIGNMENT,  # assigment
	'>=>' : TOKEN_RETURN      # return value, exit context
}

def get_cleaned_trace_node(rts):
	"""
		smoke_trace_nodes = ["    0.0003      70640   -> {main}() /var/www/MIS/secure_login.php:0",
		"    0.0009     110932     -> include_once(/u/web/dbvars.php) /var/www/MIS/secure_login.php:12",
		"                           => GLOBALS['cfg']['download']['path'] = '/tmp/' /u/web/dbvars.php:4",
		"                           >=> 1"
		]
	"""
	if TOKEN_INITIATOR not in rts:
		return False, rts, None
	else:
		init_location = rts.find(TOKEN_INITIATOR)
		previous_char = rts[init_location - 1]

		if previous_char in ('-', '=', ' '):
			PADDING_ADJUST = 2 if previous_char in ('-', '=') else 4

			returned_token = TOKEN_FCT_CALL
			if '=' == previous_char:
				returned_token = TOKEN_ASSIGNMENT
			elif ' ' == previous_char:
				returned_token = TOKEN_RETURN
			return True, rts[rts.find(previous_char + '>') + PADDING_ADJUST:], returned_token

		return False, rts, None


def clean_string_content(l, empty_string=False):
	i = 0
	state = "source"
	in_buff = ""
	com_delim = None
	max_len = len(l)
	while i < max_len:
		if state == "string":
			if l[i] == com_delim:
				if l[i-1] == '\\':
					if l[i-2] == '\\':
						in_buff += l[i]
						state = "source"
						com_delim = None
					else:
						if empty_string:
							in_buff += ' '
				else:
					in_buff += l[i]
					state = "source"
					com_delim = None
			elif l[i] == '%' and l[i-1] != '%':
				in_buff += l[i]
			else:
				if empty_string:
					in_buff += ' '
		else:
			# simply in source
			if state == "source" and (l[i] == '"' or l[i] == "'"):
				state = "string"
				com_delim = l[i]
				in_buff += l[i]
			else:
				in_buff += l[i]
		i+=1
	return in_buff


reg_simple_var = re.compile("^\$([^=]+)\s*=\s*(.+)$", re.I)
def parse_assignment(rts, delimiter='='):
	"""
		return 4 values
		@1 - Actual assignment?
		@2 - LHS if assignment, else complete data
		@3 - operator if assignment if assignment, else None
		@4 - RHS if assignment else NOne
		@5 - Type (class, etc.)
	"""
	if ' = class ' in rts:
		deli = rts.split(' = class ')
		return True, deli[0], '=', deli[1], 'class'
	elif reg_simple_var.match(rts):
		# we can simply extract the variable name with the place of the first operator
		op = rts.find('=')
		variable = rts[:op].strip()
		value = rts[op + 1:].strip()
		return True, variable, '=', value, None
	else:
		mock = clean_string_content(rts).replace('=>', '~>')
		if delimiter in mock:
			operator = None
			deli = mock.strip().split()
			if 3 <= len(deli):
				operator = deli[1]
			delo_pos = rts.find(operator)
			lhs, rhs = rts[:delo_pos], rts[delo_pos:]
			return True, lhs.strip(), operator, rhs.strip(), None
		else:
			return False, rts.strip(), None, None, None


def parse_parameters(rts):
	"""
		From a string of parametrers and values, extract a dictionnary of parameters
	"""
	mock = clean_string_content(rts, empty_string=True)
	if len(rts) != len(mock):
		raise Exception("parse_parameters:- Error with this string {%s}" % rts)
	num_parameters = mock.count(',') + 1
	if num_parameters <= 1:
		return [rts]
	else:
		params = []
		local = mock
		start = 0
		pos = 999
		while True:
			pos = local.find(',')
			if -1 == pos:
				params.append(rts[start:].strip())
				break
			p = rts[start:pos]
			params.append(p.strip())
			start = pos + 1
			local = local[start:]
		return params
	return [rts]


reg_function = re.compile("^([\$\w\d_:\->]+)\s*\((.*)\)$", re.I)
def extract_information(token, rts):
	if TOKEN_RETURN == token:
		return {'type' : 'return',  'value' : rts.replace('>=>', '').strip()}
	else:
		rts = rts.replace('\n', '').replace('\r', '').strip()
		line_number, file_name = 0, ''
		try:
			pos = rts.rfind(':')
			line_number = int(rts[pos + 1:])
			rts = rts[:pos]
			pos = rts.rfind(' ')
			file_name = rts[pos:].strip()
			rts = rts[:pos]
		except:
			line_number, file_name = 0, 'ERROR'

		if MAIN in rts:
			return {'type' : 'main', 'filename' : file_name, 'line' : line_number}
		elif TOKEN_ASSIGNMENT == token:
			# trying to find a `something = else`
			if 1 == rts.count('='):
				# easy case, just split by '='
				inf = rts.split('=')
				return {'type' : 'assignment', 'variable' : inf[0].strip(), 'value' : inf[1].strip(), 'filename' : file_name, 'line' : line_number}
			else:
				suc, lhs, op, rhs, _type = parse_assignment(rts)
				if suc:
					return {'type' : 'assignment', 'variable' : lhs.strip(), 'value' : rhs.strip(), 'filename' : file_name, 'line' : line_number}
		else:
			# function call....
			s_result = reg_function.search(rts)
			if s_result:
				s_result = s_result.groups()
				function_name = s_result[0]
				parameters = s_result[1]
				parameters = parse_parameters(parameters)
				return {'type' : 'call', 'function' : function_name, 'parameters' : parameters, 'filename' : file_name, 'line' : line_number}
		return None


# TRACE START [2010-10-06 13:10:03]
def extract_datetime(rts):
	dt = rts[rts.find('[') + 1:rts.rfind(']')]
	try:
		return time.strptime(dt, "%Y-%m-%d %H:%M:%S")
	except Exception, error:
		log.error("extract_datetime:- Error handling string [%s] (Exception: %s)" % (rts, error))
	return time.localtime()



reg_traces = re.compile("TRACE START(.+)?TRACE END", re.I)
seralizedFileName = 'model-xdbpg.pkl'
class Modeler(object):
	
	def __init__(self):
		self.times = []
		self.guid = 1
		self.cuid = 1
		self.g = digraph()
		self.context = {}
		self.rtids = {}
		self.node_location = {}
		self.accumulator = []
		self.trace_ids = []
		self.cache = {}

	def feedAccumulator(self, data):
		self.accumulator.append(data)


	@staticmethod
	def __node_uid(trace_id, node_str, cur_index=''):
		return hash(str(trace_id) + node_str + str(cur_index))

	@staticmethod
	def add_node(g, nid, location=None, linenumber=None, context_id=None, trace_id=None, method=None, data=None, _type=None, _id=None):
		if not g.has_node(nid):
			g.add_node(nid)
			g.node_attr[nid] = {'return' : '', 'context_id' : context_id, 'trace_id' : trace_id, 'location' : location, 'linenumber' : linenumber, 'method' : method, 'data' : data, 'type' : _type, 'rtid' : False, 'sequence' : _id}

	@staticmethod
	def add_edge(g, e):
		if not g.has_edge(e):
			g.add_edge(e)
		else:
			g.set_edge_weight(e, g.edge_weight(e) + 1)

	def update_rtid(self, lst):
		for e in lst:
			if e not in self.rtids:
				self.rtids[e] = re.compile(e, re.I)

	@staticmethod
	def regexp_interesect(data, dct):
		if not isinstance(data, str):
			data = str(repr(data))
		for reg in dct:
			if dct[reg].search(data):
				return True
		return False


	def update_model_taints(self):
		# get latest list of rtids
		self.update_rtid(db_list_rtid())
		
		# make sure we don't iterate if no data to check with
		if 1 > len(self.rtids):
			return
		
		# loop over the entire graph and update the 'rtid' node attr
		for n in self.g:
			if Modeler.regexp_interesect(self.g.node_attr[n]['method'], self.rtids) or Modeler.regexp_interesect(self.g.node_attr[n]['data'], self.rtids) or Modeler.regexp_interesect(self.g.node_attr[n]['return'], self.rtids):
				# update taint
				self.g.node_attr[n]['rtid'] = True

	def processTrace(self, trace, trace_id):
		callstack = Stack()
		prev_node, cur_node = None, None
		pre_l = None
		cur_i = 0
		
		if trace_id not in self.trace_ids:
			self.trace_ids.append(trace_id)
		
		for l in trace:
			self.guid += 1
			cur_i += 1
			suc, trace, token = get_cleaned_trace_node(l)
			if not suc:
				continue
			else:
				trace_info = extract_information(token, trace)
				if trace_info and isinstance(trace_info, dict):
					_type = trace_info['type']
					
					if _type == 'main':
						main_nuid = Modeler.__node_uid(trace_id, l)
						Modeler.add_node(self.g, main_nuid, trace_info['filename'], trace_info['line'], None, trace_id, 'main', self.guid)
						callstack.push(main_nuid)
					elif _type == 'return':
						# only keep the latest  value
						if not callstack.empty():
							prev_node = callstack.pop()
							self.g.node_attr[prev_node]['return'] = trace_info['value']
							
					elif _type == 'call':
						cur_node = Modeler.__node_uid(trace_id, l)
						Modeler.add_node(self.g, cur_node, trace_info['filename'], trace_info['line'], None, trace_id, trace_info['function'], trace_info['parameters'], 'call', self.guid)
						if not callstack.empty():
							prev_node = callstack.top()
							edge = (prev_node, cur_node)
							Modeler.add_edge(self.g, edge)
							# set label to the edge as parameters
							self.g.set_edge_label(edge, ','.join(trace_info['parameters']))
						callstack.push(cur_node)

					elif _type == 'assignment':
						cur_node = Modeler.__node_uid(trace_id, l)
						Modeler.add_node(self.g, cur_node, trace_info['filename'], trace_info['line'], None, trace_id, trace_info['variable'], trace_info['value'], 'assignment', self.guid)

						if not callstack.empty():
							prev_node = callstack.top()
							edge = (prev_node, cur_node)
							Modeler.add_edge(self.g, edge)
							
							# set label to the edge as parameters
							self.g.set_edge_label(edge, trace_info['value'])
						# callstack.push(cur_node)

		self.update_model_taints()
		# self.write_dot_slice(trace_id)

	def update_JSON(self):
		log.debug("update_JSON")
		# only output simple paths and possible ways to investigate
		jsonified_data = {'injected' : self.rtids.keys(), 'hash' : '', 'trace' : {}, 'knowledge' : {'function' : {}, 'file' : {}}}

		def safe_unicode(obj, *args):
			try:
				return unicode(obj, *args)
			except UnicodeDecodeError:
				# obj is byte string
				ascii_text = str(obj).encode('string_escape')
				return unicode(ascii_text)
		
		def lmt_size(rts):
			length = len(rts)
			if 256 >= length:
				return rts
			return rts[:256]

		# always output the top 50 traces
		sorted_traces = self.trace_ids
		sorted_traces.sort(reverse=True)
		
		for trace_id in sorted_traces[:min(100, len(sorted_traces))]:
			if trace_id in jsonified_data['trace']:
				continue

			jsonified_data['trace'][trace_id] = {
				'callchain' : [],
				'function' : {},
				'file' : {},
			}

			tlst_nodes = {}
			for n in self.g:
				if trace_id == self.g.node_attr[n]['trace_id']:
					tlst_nodes[self.g.node_attr[n]['sequence']] = n
			slst_nodes = [tlst_nodes[k] for k in sorted(tlst_nodes.keys())]
			
			if 0 == len(slst_nodes):
				del jsonified_data['trace'][trace_id]
			else:
				for n in slst_nodes:
					if self.g.node_attr[n]['type'] not in ('call', 'main'):
						continue
					
					if self.g.node_attr[n]['rtid']:
						nd = self.g.node_attr[n]
						method = safe_unicode(nd['method'])
						data = safe_unicode(nd['data'])
						ret = safe_unicode(nd['return'])

						if	has_blacklisted_function(nd['location']):
							continue

						jsonified_data['trace'][trace_id]['callchain'].append([self.g.node_attr[n]['type'], nd['location'], nd['linenumber'], method, lmt_size(data), lmt_size(ret)])
						
						"""
						if method not in jsonified_data['trace'][trace_id]['function']:
							jsonified_data['trace'][trace_id]['function'][method] = []
						jsonified_data['trace'][trace_id]['function'][method].append(data)

						if method not in jsonified_data['knowledge']['function']:
							jsonified_data['knowledge']['function'][method] = []
						if data not in jsonified_data['knowledge']['function'][method]:
							jsonified_data['knowledge']['function'][method] = data

						if nd['location'] not in jsonified_data['trace'][trace_id]['file']:
							jsonified_data['trace'][trace_id]['file'][nd['location']] = []
						jsonified_data['trace'][trace_id]['file'][nd['location']].append(nd['linenumber'])
						"""
						
				if 0 == len(jsonified_data['trace'][trace_id]['callchain']):
					del jsonified_data['trace'][trace_id]

		jsonified_data['hash'] = hash(str(repr(jsonified_data['trace']) + repr(jsonified_data['injected'])))

		o = open('assmnt-data.json', 'w')
		o.write(json.dumps(jsonified_data, ensure_ascii=False, separators=(',', ':') ))
		o.close()


	def write_dot_slice(self, trace_id):
		new_g = digraph()
		new_g.add_graph(self.g)

		# create new graph with only the CWE instances
		for n in new_g:
			if self.g.node_attr[n]['trace_id'] != trace_id:
				new_g.del_node(n)
				continue

		fname = './graphs/%s-trace' % str(trace_id)
		self.write_dot(new_g, fname)
		print "Generated: %s" % fname

	def write_dot(self, g, fname):
		if 1 > len(g):
			return

		def pen_width(num, local_max):
			max_penwidth = 7
			val = int(math.ceil(float(num) / float(local_max) * float(max_penwidth)))
			return max(1, min(val, max_penwidth))

		def node_call_labels(dct):
			"""
				<f0> Function Name| $value1 | ... | return values"
			"""
			lbl_str = "%s:%s |" % (dct['location'], dct['linenumber'])
			lbl_str +=  "<f0> " + xml_entities(dct['method']) + "()"
			for e in dct['data']:
				lbl_str += ' | ' + xml_entities(e)
			lbl_str += '| <f1> T_RET: ' + xml_entities(dct['return'])
			return lbl_str

		# generate DOT file with custom information (source, sink, and custom colors)
		dstr = """digraph runtime {
			graph [rankdir = "LR"];
			node [fontsize = 12];
		"""
		for n in g:
			node_type = 'shape=ellipse'
			if self.g.node_attr[n]['type'] == 'call':
				node_type = 'shape=record'
				node_type += ", label=\"%s\"" % node_call_labels(self.g.node_attr[n])
			elif self.g.node_attr[n]['type'] == 'assignment':
				node_type = 'shape=record, style="filled,bold" penwidth=1'
				node_type += ",label=\"%s=%s\"" % (xml_entities(self.g.node_attr[n]['method']), xml_entities(self.g.node_attr[n]['data']))
			else:
				if self.g.node_attr[n]['method'] == 'return':
					node_type += ", label=\"T_RETURN:%s\"" % self.g.node_attr[n]['data'].replace('"', '\\"')
				else:
					node_type += ", label=\"%s\"" % self.g.node_attr[n]['method']
					
			if self.g.node_attr[n]['rtid']:
				node_type += ', fillcolor="#AA0114"'

			#if 'source' in g.node_attr[n]['nodetype'] or 'sink' in g.node_attr[n]['nodetype']:
			#	node_type = 'shape=box, style=filled, fillcolor="%s"' % color_node(g.node_attr[n]['nodetype'])
			# get the size of the pen based on the number of GUID
			#node_type += ', penwidth=%d' % int(pen_width(len(g.node_attr[n]['guid']), max_guid))
			dstr += '  "%s" [%s];\n' % (n, node_type)

		for e in g.edges():
			n1, n2 = e
			node_type = ""
			#l_e_prop = self.g.edge_label(e).split(',')
			if self.g.node_attr[n1]['rtid'] and self.g.node_attr[n2]['rtid']:
				node_type += 'color="#AA0114"'
			else:
				node_type += 'color="black"'
			node_type += ', penwidth=%d' % pen_width(g.edge_weight(e), 10)
			
			#node_type += ', label="%s"' %  ','.join(l_e_prop).replace('"', '\\"')
			addon_n1 = ':f0' if self.g.node_attr[n1]['type'] in ('call', 'return') else ''
			addon_n2 = ':f0' if self.g.node_attr[n2]['type'] in ('call', 'return') else ''
			dstr += ' "%s"%s -> "%s"%s [%s];\n' % (n1, addon_n1, n2, addon_n2, node_type)
		dstr += '}'

		dot_fname = fname + '.dot'
		o = open(dot_fname, 'w')
		o.write(dstr)
		o.close()

		#cmd_line = "dot %s -Tsvg -O -Kdot -x" % (dot_fname)
		# os.system(cmd_line)
		syscall("dot", [dot_fname, "-Tsvg", "-O", "-Kdot", "-x"])
		

	def export(self):
		return
		if 0 < len(self.g):
			self.write_dot(self.g, './graphs/global_file')

	def run(self):

		def safe_unicode(obj, *args):
			try:
				return unicode(obj, *args)
			except UnicodeDecodeError:
				# obj is byte string
				ascii_text = str(obj).encode('string_escape')
				return unicode(ascii_text)

		
		def get_appname(rts):
			rts = rts.lower()
			for name in ('paypage', 'ccentry', 'mis'):
				if name in rts:
					return name
			return 'UnknownEntryPoint'


		log.debug("Execute model checks and execution of loop")
		# process the accumulated data into the model
		bl_trace_id = []
		traces, last_trace_end_str = {}, None
		trace_id = self.guid
		start_trace = False
		local = []
		latest_datetime = None
		try:
			for line in self.accumulator:
				if not start_trace:
					if 'TRACE START' == line[:11]:
						start_trace = True
						latest_datetime = extract_datetime(line)
						local = []
				else:
					if 'TRACE END' in line[:9]:
						appname = get_appname(local[0])
						trace_id = time.strftime("%Y-%m-%d_%H-%M-%S", latest_datetime) + '_' + appname + '_' + str(hex(hash(repr(local))))
						if trace_id not in bl_trace_id:
							bl_trace_id.append(trace_id)
							last_trace_end_str = line
							start_trace = False
							traces[trace_id] = local
	
							if trace_id not in self.cache:
								self.cache[trace_id] = local
							
						local = []
						start_trace = False
					else:
						local.append(line)

			# keep only what hasn't been processed yet
			if last_trace_end_str in self.accumulator:
				self.accumulator = self.accumulator[self.accumulator.index(last_trace_end_str):]
	
			log.debug("Process all %d traces available" % len(traces))
			for trace_id in traces:
				self.processTrace(traces[trace_id], trace_id)
				try:
					o = open('./cache/' + trace_id + '.html', 'w')
	
					o.write(pygment_content("<?php\n"  + '\n'.join([safe_unicode(rts) for rts in self.cache[trace_id] if not has_blacklisted_function(rts)]) + "\n?>"))
					o.close()
				except IOError, error:
					log.error("Exception writing %s (error:%s)" % ('./cache/' + trace_id + '.html', error))
			
			# generate the JSON file for frontend to consume
			self.update_JSON()
		except Exception, error:
			log.error("run- %s" % error)
			sys.exit()

	
	def hasEnoughData(self):
		if 0 == len(self.accumulator):
			return False
		if reg_traces.search('|'.join(self.accumulator)):
			return True
		return False


def process_files(conf):
	global USER, HOST, CMD
	USER, HOST = conf['username'], conf['host']
	# check for the SCP command
	
	start_time = time.time()
	CMD = 'tail -f -n 1 ' + conf['remote_location']
	create_db()
	protocol.ClientCreator(reactor, Transport).connectTCP(HOST, 22)
	reactor.run()
	if time.time() - start_time < 30:
		print "spent less than 30s? maybe the file isn't ready yet... we add data manually then."
		CMD = 'echo "RTDI_DATA" >> ' + conf['remote_location']
		protocol.ClientCreator(reactor, Transport).connectTCP(HOST, 22)
		reactor.run()
		CMD = 'tail -f -n 1 ' + conf['remote_location']
		protocol.ClientCreator(reactor, Transport).connectTCP(HOST, 22)
		reactor.run()

""" TESTING CODE
fname = 'test_tr_1.txt' #'xdbg-trace.2043925204.xt' #
luffer = open(fname, 'r').readlines()
luffer += luffer * 10
model = Modeler()
for l in luffer:
	model.feedAccumulator(l)
model.run()
# model.export()
"""

def main(argc, argv):
	conf = {
		'host' : None,
		'username' : None,
		'password' : None,
		'remote_location' : None,
		'remote_filter' : None
	}

	for i in range(argc):
		s = argv[i]
		if s in ('--host', '-h'):
			conf['host'] = __normalize_argmt(argv[i+1])
		elif s in ('--username', '-u'):
			conf['username'] = argv[i+1]
		elif s in ('--password', '-p'):
			conf['password'] = argv[i+1]
		elif s in ('--remote', '-r'):
			conf['remote_location'] = argv[i+1]
		elif s in ('--filter', '-f'):
			conf['filter'] = argv[i+1]

	if conf['host']:
		process_files(conf)

if __name__ == "__main__":
	main(len(sys.argv), sys.argv)