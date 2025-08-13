import zipfile
wheel = r'C:\Users\Diogo Novo\Documents\GitHub\nseekfs\target\wheels\nseekfs-0.1.0-cp311-cp311-win_amd64.whl'
print('Checking wheel contents:')
with zipfile.ZipFile(wheel, 'r') as z:
    files = z.namelist()
    for f in sorted(files):
        print(f'  {f}')
    
    pyd_files = [f for f in files if f.endswith('.pyd')]
    if pyd_files:
        print(f'\n✅ SUCCESS: Rust module found: {pyd_files}')
    else:
        print(f'\n❌ FAIL: Still no .pyd in wheel')
