"""Enable only native measured-current limiting while disarmed and in Manual."""
import argparse
import math
from pathlib import Path
import subprocess
import time

PARAMS = (('MOT_BAT_CURR_MAX', 15.0, 9), ('MOT_BAT_CURR_TC', 0.5, 9))


def main():
    from pymavlink import mavutil
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',required=True)
    parser.add_argument('--max-current',type=float,default=15.0)
    parser.add_argument('--time-constant',type=float,default=0.5)
    args=parser.parse_args()
    if not math.isfinite(args.max_current) or not 5 <= args.max_current <= 25:
        raise RuntimeError('Current limit must be 5..25 A')
    if not math.isfinite(args.time_constant) or not .1 <= args.time_constant <= 10:
        raise RuntimeError('Time constant must be 0.1..10 s')
    targets=(('BATT_MONITOR',4.0,2),('BATT_AMP_PERVLT',None,9),
             ('MOT_BAT_CURR_MAX',args.max_current,9),('MOT_BAT_CURR_TC',args.time_constant,9))
    port=str(Path(args.port).resolve(strict=True))
    owner=subprocess.run(['fuser',port],capture_output=True,text=True,timeout=5)
    if owner.returncode != 1 or owner.stdout.strip():
        raise RuntimeError('Serial port is not free')
    link=mavutil.mavlink_connection(port,baud=115200,source_system=254,source_component=191)
    try:
        link.port.exclusive=True
        hb=link.wait_heartbeat(timeout=10)
        if hb is None or hb.type != mavutil.mavlink.MAV_TYPE_SUBMARINE or hb.base_mode & 128 or hb.custom_mode != 19:
            raise RuntimeError('Require fresh disarmed ArduSub in Manual; no writes')
        def read(name):
            link.mav.param_request_read_send(1,1,name.encode('ascii'),-1)
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                msg=link.recv_match(blocking=True,timeout=.2)
                if msg is None or msg.get_srcSystem()!=1 or msg.get_srcComponent()!=1: continue
                if msg.get_type()=='HEARTBEAT' and (msg.base_mode & 128 or msg.custom_mode != 19):
                    raise RuntimeError('Vehicle state changed; abort')
                if msg.get_type()=='PARAM_VALUE':
                    name_in=msg.param_id.decode('ascii') if isinstance(msg.param_id,bytes) else msg.param_id
                    if name_in.rstrip('\x00')==name: return float(msg.param_value),int(msg.param_type)
            raise RuntimeError('No readback: '+name)
        before={}
        for name,target,kind in targets:
            value,actual_kind=read(name)
            if actual_kind!=kind or not math.isfinite(value): raise RuntimeError('Invalid parameter: '+name)
            if name=='BATT_MONITOR' and value!=target: raise RuntimeError('PX4 power monitor is not enabled')
            if name=='BATT_AMP_PERVLT' and value<=0: raise RuntimeError('Current-sensor scale is invalid')
            before[name]=value
        for name,target,kind in targets[2:]:
            if not math.isclose(before[name],target,abs_tol=1e-7):
                hb=link.recv_match(type='HEARTBEAT',blocking=True,timeout=3)
                if hb is None or hb.get_srcSystem()!=1 or hb.get_srcComponent()!=1 or hb.base_mode & 128 or hb.custom_mode!=19:
                    raise RuntimeError('No fresh disarmed Manual heartbeat before write')
                link.mav.param_set_send(1,1,name.encode('ascii'),target,kind)
                time.sleep(.3)
            value,actual_kind=read(name)
            if actual_kind!=kind or not math.isclose(value,target,abs_tol=1e-7):
                raise RuntimeError('Independent readback mismatch: '+name)
            print(f'VERIFIED {name}={value:g}',flush=True)
    finally:
        link.close()


if __name__=='__main__': main()
