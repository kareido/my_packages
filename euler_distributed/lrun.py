import argparse
import os
import socket
import subprocess
   
        
def _get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

        
def lrun(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntasks', '-n', type=int, required=True)
    parser.add_argument('cmd', nargs=argparse.REMAINDER)
    opt = parser.parse_args()
    cmd = ' '.join(opt.cmd)

    proc_list = []
    for proc_id in range(opt.ntasks):
        env = os.environ.copy()
        env['SLURM_NTASKS'] = str(opt.ntasks)
        env['SLURM_PROCID'] = str(proc_id)
        env['SLURM_NODELIST'] = '10086 ' + _get_host_ip().replace('.', '-')
        proc_list.append(subprocess.Popen(cmd, shell=True, env=env))


