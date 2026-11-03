import re,hashlib,os,sys
man,base_lvr,repo=sys.argv[1:4]
txt=open(man).read()
rows=re.findall(r'^\s*\|[^\n]*?\[`([^`]+)`\]\(file://[^)]*?/external_models/lvr/([^)#]+)[^)]*\)\s*\|\s*([^|]*)\|\s*([^|]*)\|\s*([^|]*)\|\s*`([0-9a-f]{64})`',txt,re.M)
ok=bad=miss=0
for disp,path,wc,nl,by,sha in rows:
    if path.startswith('repo/'): fp=os.path.join(repo,path[5:])
    else: fp=os.path.join(base_lvr,path)
    if not os.path.exists(fp): miss+=1; print('MISSING',path); continue
    data=open(fp,'rb').read()
    h=hashlib.sha256(data).hexdigest()
    errs=[]
    if h!=sha: errs.append('sha')
    b=by.strip().replace(',','')
    if b.isdigit() and int(b)!=len(data): errs.append(f'bytes {b}!={len(data)}')
    w=wc.strip().replace(',','')
    if w.isdigit():
        newlines = data.count(b'\n')
        if int(w) != newlines:
            errs.append(f'wc {w}!={newlines}')
    n=nl.strip().replace(',','')
    if n.isdigit():
        real=data.count(b'\n')+(0 if data.endswith(b'\n') or not data else 1)
        if int(n)!=real: errs.append(f'nl {n}!={real}')
    if errs: bad+=1; print('MISMATCH',path,errs)
    else: ok+=1
print('rows',len(rows),'ok',ok,'bad',bad,'missing',miss)
