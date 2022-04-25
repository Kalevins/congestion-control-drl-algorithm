import subprocess

def setReceiver():
    subprocess.call(['sudo','docker','exec','-it','mn.d2','./ITGRecv'])

def sendTraffic():
    #Prueba de tráfico de una cirugía remota
    subprocess.run(['sudo','docker','exec','-it','mn.d1','./ITGSend','-a','10.0.0.252','-rp','10001','-C','10000','-c','20000','-T','UDP','-t','10000','-l','sender.log','-x','receiver.log','&'],shell=True)
    subprocess.run(['sudo','docker','exec','-it','mn.d1','./ITGSend','-a','10.0.0.252','-rp','10002','-C','10000','-c','20000','-T','UDP','-t','10000','-l','sender2.log','-x','receiver2.log'])
    
sendTraffic()
