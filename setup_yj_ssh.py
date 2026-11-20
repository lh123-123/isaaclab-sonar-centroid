import pexpect
import sys

REMOTE_SSH = "ssh -p 8001 -o StrictHostKeyChecking=no lh@47.101.175.205"
SUDO_PASS = "usstrobot123"

print("Connecting to remote server...")
child = pexpect.spawn(REMOTE_SSH, encoding='utf-8', timeout=30)
child.logfile = sys.stdout

# Wait for password prompt (if any) or shell prompt
try:
    child.expect(r'\$|sudo.password:', timeout=15)
except:
    pass

# If we got a password prompt from SSH key auth, send nothing (keys work)
# Just wait for shell prompt
child.expect(r'\$')

print("\n--- Sending sudo commands ---\n")

commands = [
    f"sudo -S mkdir -p /home/yj/.ssh",
    f"echo {SUDO_PASS} | sudo -S tee /home/yj/.ssh/authorized_keys",
    "sudo chown -R yj:yj /home/yj/.ssh",
    "sudo chmod 700 /home/yj/.ssh",
    "sudo chmod 600 /home/yj/.ssh/authorized_keys",
    "echo DONE",
]

for cmd in commands:
    child.sendline(cmd)
    idx = child.expect([r'\[sudo\] password.*:', r'DONE', r'\$', pexpect.TIMEOUT], timeout=10)
    if idx == 0:
        print(f"[SUDO PROMPT] Sending password for: {cmd[:40]}...")
        child.sendline(SUDO_PASS)
    elif idx == 1:
        print("Setup complete!")
        break
    elif idx == 2:
        print(f"[OK] {cmd[:50]}")
    else:
        print(f"[TIMEOUT] {cmd[:50]}")

child.sendline("exit")
child.close()
