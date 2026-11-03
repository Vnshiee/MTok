import zipfile,pickle,sys,io
class Stub:
    def __init__(self,*a,**k): self.args=a
    def __setstate__(self,s): self.state=s
    def __call__(self,*a,**k): return Stub(*a)
class SafeU(pickle.Unpickler):
    def find_class(self,mod,name):
        return type(f"{mod}.{name}",(Stub,),{})
    def persistent_load(self,pid): return ('persistent',pid)
f=sys.argv[1]
if zipfile.is_zipfile(f):
    z=zipfile.ZipFile(f); n=[x for x in z.namelist() if x.endswith('data.pkl')][0]; data=z.read(n)
else: data=open(f,'rb').read()
o=SafeU(io.BytesIO(data)).load()
st=o.state if hasattr(o,'state') else o.__dict__
print(type(o).__name__)
keys=sys.argv[2:]
for k in sorted(st):
    if not keys or any(q in k for q in keys):
        v=st[k]
        if isinstance(v,Stub): v=f"<{type(v).__name__} {getattr(v,'args',None)} {getattr(v,'state','')}>"
        print(f"{k} = {v!r}"[:200])
