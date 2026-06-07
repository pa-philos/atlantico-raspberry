from zeroconf import ServiceBrowser, Zeroconf, ServiceInfo
import socket
import time

# O ID que configuramos no Avahi do servidor
SERVER_PROP_KEY = b'server_id'
TARGET_SERVER_ID = b"federative_server" 

class MyListener:
    def __init__(self):
        self.found_ip = None
        self.found_port = None

    def remove_service(self, zeroconf, type, name):
        pass

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        
        if info:
            props = info.properties 
            
            if props.get(SERVER_PROP_KEY) == TARGET_SERVER_ID:
                
                # 2. Converter o endereço IP (que vem em bytes) para string
                # O zeroconf pode retornar múltiplos endereços, pegamos o primeiro
                if info.addresses:
                    self.found_ip = socket.inet_ntoa(info.addresses[0])
                    self.found_port = info.port
        
    def update_service(self, zeroconf, type, name):
        pass

zeroconf = Zeroconf()
listener = MyListener()
browser = ServiceBrowser(zeroconf, "_mqtt._tcp.local.", listener)

def clear_discovery():
    global zeroconf, browser, listener
    if zeroconf:
        zeroconf.close()
    if browser:
        browser.cancel()
    zeroconf = None
    browser = None
    listener = None