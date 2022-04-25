import subprocess

for i in range(1,33):    
    subprocess.call(['ITGSend','-T','UDP','-a','10.0.0.23','-z','1000','-c','100','-t','600000','-l','/home/ubuntu/Desktop/DITG/senders/sender_'+str(i)+'.log','-x','/home/ubuntu/Desktop/DITG/receivers/receiver_'+str(i)+'.log'])
