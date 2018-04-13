import json, glob, os

data_base = {}
os.chdir("./faces/")

sum = 0
for file in glob.glob('*.attr'):
    f = open(file, 'r')
    data_base[file[:-5]] = json.loads(f.read());
    sum += 1

f = open('../face_database.txt', 'w');
f.write(json.dumps(data_base))
f.close()

print('total: ' + str(sum))