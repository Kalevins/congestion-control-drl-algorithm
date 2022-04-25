import os


def sendTraffic():
    #Prueba de tráfico de una cirugía remota
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10001 -C 10000 -c 20000 -T UDP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10002 -C 3 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10003 -C 49 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10004 -C 375 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10005 -C 32 -c 1568 -T UDP -t 10000 -x receiver.log &')

def addSurgery():
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10006 -C 10000 -c 20000 -T UDP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10007 -C 3 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10008 -C 49 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10009 -C 375 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10010 -C 32 -c 1568 -T UDP -t 10000 -x receiver.log &')

sendTraffic()
addSurgery()
