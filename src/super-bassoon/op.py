import subprocess
#from onepassword import Client

def get_secret(reference: str) -> str:
    result = subprocess.run(["op", "read", reference], capture_output=True, text=True, check=True)
    return result.stdout.strip()
