from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

numCoresD1 = 4
numCoresD2 = 1
resources = "cpu"

data1 = np.genfromtxt('results/'+str(numCoresD1)+'_'+str(numCoresD2)+'_'+resources+'_'+str(1)+'RS'+'.csv', delimiter=',')
data2 = np.genfromtxt('results/'+str(numCoresD1)+'_'+str(numCoresD2)+'_'+resources+'_'+str(2)+'RS'+'.csv', delimiter=',')
data3 = np.genfromtxt('results/'+str(numCoresD1)+'_'+str(numCoresD2)+'_'+resources+'_'+str(3)+'RS'+'.csv', delimiter=',')
data4 = np.genfromtxt('results/'+str(numCoresD1)+'_'+str(numCoresD2)+'_'+resources+'_'+str(4)+'RS'+'.csv', delimiter=',')

yLatency1 = data1[0]
yPacketLost1 = data1[1]
yJitter1 = data1[2]
yUsageResources1 = data1[3]
yAsignedResources1 = data1[4]

yLatency2 = data2[0]
yPacketLost2 = data2[1]
yJitter2 = data2[2]
yUsageResources2 = data2[3]
yAsignedResources2 = data2[4]

yLatency3 = data3[0]
yPacketLost3 = data3[1]
yJitter3 = data3[2]
yUsageResources3 = data3[3]
yAsignedResources3 = data3[4]

yLatency4 = data4[0]
yPacketLost4 = data4[1]
yJitter4 = data4[2]
yUsageResources4 = data4[3]
yAsignedResources4 = data4[4]

# Grafica latencia
plt.xlabel("Episodios")
plt.ylabel("Latencia [ms]")
plt.ylim(0, 0.12)
plt.plot(range(len(yLatency1)), yLatency1, color = 'red', label = '1 Cirugía Remota')
plt.plot(range(len(yLatency2)), yLatency2, color = 'blue', label = '2 Cirugías Remotas')
plt.plot(range(len(yLatency3)), yLatency3, color = 'green', label = '3 Cirugías Remotas')
plt.plot(range(len(yLatency4)), yLatency4, color = 'orange', label = '4 Cirugías Remotas')
plt.legend()
plt.grid()
plt.savefig('graphs/'+str(numCoresD1)+'_'+str(numCoresD2)+'_'+resources+'_latency'+'.pdf')
plt.show()
# Grafica paquetes perdidos
plt.xlabel("Episodios")
plt.ylabel("Paquetes perdidos [%]")
plt.ylim(0, 2)
plt.plot(range(len(yPacketLost1)), yPacketLost1, color = 'red', label = '1 Cirugía Remota')
plt.plot(range(len(yPacketLost2)), yPacketLost2, color = 'blue', label = '2 Cirugías Remotas')
plt.plot(range(len(yPacketLost3)), yPacketLost3, color = 'green', label = '3 Cirugías Remotas')
plt.plot(range(len(yPacketLost4)), yPacketLost4, color = 'orange', label = '4 Cirugías Remotas')
plt.legend()
plt.grid()
plt.savefig('graphs/'+str(numCoresD1)+'_'+str(numCoresD2)+'_'+resources+'_packetLost'+'.pdf')
plt.show()
# Grafica jitter
plt.xlabel("Episodios")
plt.ylabel("Jitter [ms]")
plt.ylim(0, 0.1)
plt.plot(range(len(yJitter1)), yJitter1, color = 'red', label = '1 Cirugía Remota')
plt.plot(range(len(yJitter2)), yJitter2, color = 'blue', label = '2 Cirugías Remotas')
plt.plot(range(len(yJitter3)), yJitter3, color = 'green', label = '3 Cirugías Remotas')
plt.plot(range(len(yJitter4)), yJitter4, color = 'orange', label = '4 Cirugías Remotas')
plt.legend()
plt.grid()
plt.savefig('graphs/'+str(numCoresD1)+'_'+str(numCoresD2)+'_'+resources+'_jitter'+'.pdf')
plt.show()
# Grafica recursos
ylabels = {'cpu': "CPU [Deci-Unidades]", 'mem': "Memoria [MB]"}
plt.xlabel("Episodios")
plt.ylabel(ylabels[resources])
plt.ylim(0, 80)
plt.plot(range(len(yUsageResources1)), yUsageResources1, color = 'red', linestyle = 'dashed', label = '1 Cirugía Remota')
plt.plot(range(len(yUsageResources2)), yUsageResources2, color = 'blue', linestyle = 'dashed', label = '2 Cirugías Remotas')
plt.plot(range(len(yUsageResources3)), yUsageResources3, color = 'green', linestyle = 'dashed', label = '3 Cirugías Remotas')
plt.plot(range(len(yUsageResources4)), yUsageResources4, color = 'orange', linestyle = 'dashed', label = '4 Cirugías Remotas')
plt.plot(range(len(yAsignedResources1)), yAsignedResources1, color = 'red', label = '1 Cirugías Remotas')
plt.plot(range(len(yAsignedResources2)), yAsignedResources2, color = 'blue', label = '2 Cirugías Remotas')
plt.plot(range(len(yAsignedResources3)), yAsignedResources3, color = 'green', label = '3 Cirugías Remotas')
plt.plot(range(len(yAsignedResources4)), yAsignedResources4, color = 'orange', label = '4 Cirugías Remotas')

redPoint = Line2D([0], [0], marker='o', color='w', label='1 Cirugía Remota', markerfacecolor='red', markersize=10)
bluePoint = Line2D([0], [0], marker='o', color='w', label='2 Cirugías Remotas', markerfacecolor='blue', markersize=10)
greenPoint = Line2D([0], [0], marker='o', color='w', label='3 Cirugías Remotas', markerfacecolor='green', markersize=10)
orangePoint = Line2D([0], [0], marker='o', color='w', label='4 Cirugías Remotas', markerfacecolor='orange', markersize=10)
lineAsigned = Line2D([0], [0], label='Recursos Asigandos', color='black')
lineUsed = Line2D([0], [0], label='Recursos Usados', color='black', linestyle = 'dashed')

plt.legend(handles=[lineAsigned, lineUsed, redPoint, bluePoint, greenPoint, orangePoint])
plt.grid()
plt.savefig('graphs/'+str(numCoresD1)+'_'+str(numCoresD2)+'_'+resources+'_resources'+'.pdf', orientation='landscape')
plt.show()