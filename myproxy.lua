local fh = io.open("/var/log/mysql/proxy.query.log", "a+")
fh:setvbuf('line',4096)
local the_query = '';
local seqno = 0;

function read_query(packet)
    if string.byte(packet) == proxy.COM_QUERY then
        -- query = string.sub(packet, 2)
        seqno = seqno + 1
        the_query = (string.gsub(string.gsub(string.sub(packet, 2), "%s%s*", ' '), "^%s*(.-)%s*$", "%1"))
        fh:write(string.format("%s %09d %09d : %s (%s) -- %s\n",
            os.date('%Y-%m-%d %H:%M:%S'),
            proxy.connection.server.thread_id,
            seqno,
            proxy.connection.client.username,
            proxy.connection.client.default_db,
            the_query))
        fh:flush()
        proxy.queries:append(1, packet, {resultset_is_needed = true} )
        return proxy.PROXY_SEND_QUERY
    end
end

function read_query_result (inj)
    local res = assert(inj.resultset)
    -- if res.query_status == proxy.MYSQLD_PACKET_ERR then
    if (res.query_status == proxy.MYSQLD_PACKET_ERR) or (res.warning_count > 0) then
        local query = string.sub(inj.query, 2)
        local err_code     = res.raw:byte(2) + (res.raw:byte(3) * 256)
        local err_sqlstate = res.raw:sub(5, 9)
        local err_msg      = res.raw:sub(10)

        print("Query Received -\027[01;31m\027[K", query, "\027[m\027[K")
        print("Query Error code -", err_code)
        print("Query Error Sqlstate -", err_sqlstate)
        print("Query Error message -", err_msg)
        print("Query warnings -", res.warning_count)

    end
end
