import subprocess

while True:    
    subprocess.call(['iperf3','-c','10.0.0.23'])
