import subprocess
import matplotlib.pyplot as pl
import numpy as np
lista_10M = []
lista_100M = []
lista_1000M = []
for i in range(1,33):
    subprocess.call(['/usr/local/bin/ITGDec','/home/mininet/Desktop/Enfasis/10M/receiver_10M_'+str(i)+'.log','-b','600000','bw_10M_'+str(i)+'.dat'])
    subprocess.call(['/usr/local/bin/ITGDec','/home/mininet/Desktop/Enfasis/100M/receiver_100M_'+str(i)+'.log','-b','600000','bw_100M_'+str(i)+'.dat'])
    subprocess.call(['/usr/local/bin/ITGDec','/home/mininet/Desktop/Enfasis/1000M/receiver_1000M_'+str(i)+'.log','-b','600000','bw_1000M_'+str(i)+'.dat'])
    M10 = open('bw_10M_'+str(i)+'.dat','r')
    M100 = open('bw_100M_'+str(i)+'.dat','r')
    M1000 = open('bw_1000M_'+str(i)+'.dat','r')
    lineas_10M = M10.readlines()
    lineas_100M = M100.readlines()
    lineas_1000M = M1000.readlines()
    linea_datos_10M = lineas_10M[1]
    linea_datos_100M = lineas_100M[1]
    linea_datos_1000M = lineas_1000M[1]
    dato_10M = linea_datos_10M.split(' ') 
    dato_100M = linea_datos_100M.split(' ') 
    dato_1000M = linea_datos_1000M.split(' ') 
    lista_10M.append(float(dato_10M[1]))
    lista_100M.append(float(dato_100M[1]))
    lista_1000M.append(float(dato_1000M[1]))
    subprocess.call(['sudo','rm','bw_10M_'+str(i)+'.dat']) 
    subprocess.call(['sudo','rm','bw_100M_'+str(i)+'.dat']) 
    subprocess.call(['sudo','rm','bw_1000M_'+str(i)+'.dat']) 

desviacion_10M = np.std(lista_10M)
desviacion_100M = np.std(lista_100M)
desviacion_1000M = np.std(lista_1000M)
media_10M = np.mean(lista_10M)
media_100M = np.mean(lista_100M)
media_1000M = np.mean(lista_1000M)
resultado_10M = (desviacion_10M/media_10M)*100
resultado_100M = (desviacion_100M/media_100M)*100
resultado_1000M = (desviacion_1000M/media_1000M)*100

print resultado_10M
print resultado_100M
print resultado_1000M

pl.figure()
pl.plot(lista_10M,label='10M')
pl.xlabel('Medidas')
pl.ylabel('Ancho de banda [Kbit/s]')
pl.savefig('bw_10M.png',format='png')

pl.figure()
pl.plot(lista_100M,label='100M')
pl.xlabel('Medidas')
pl.ylabel('Ancho de banda [Kbit/s]')
pl.savefig('bw_100M.png',format='png')

pl.figure()
pl.plot(lista_1000M,label='1000M')
pl.xlabel('Medidas')
pl.ylabel('Ancho de banda [Kbit/s]')
pl.savefig('bw_1000M.png',format='png')

pl.figure()
pl.plot(lista_10M,label='10M')
pl.plot(lista_100M,label='100M')
pl.plot(lista_1000M,label='1000M')
pl.xlabel('Medidas')
pl.ylabel('Ancho de banda [Kbit/s]')
pl.savefig('bws.png',format='png')