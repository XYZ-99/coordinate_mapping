import os

src_folder = '7scenes_source'
focal_length = 525.0

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

print("----------------- Downloading Initialization Files -----------------")

os.system('wget -O 7scenes_init.tar.gz https://heidata.uni-heidelberg.de/api/access/datafile/:persistentId?persistentId=doi:10.11588/data/GSJE9D/RG00C4')
os.system('tar -xvzf 7scenes_init.tar.gz')
os.system('rm 7scenes_init.tar.gz')

mkdir(src_folder)
os.chdir(src_folder)

for dataset in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']:
    
    print("-------------- Downloading 7scenes Data:", dataset, "----------------")

    os.system('wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/' + dataset + '.zip')
    os.system('unzip ' + dataset + '.zip')
    os.system('rm ' + dataset + '.zip')
    
    sequences = os.listdir(dataset)

    for file in sequences:
        if file.endswith('.zip'):
            print("Unpacking", file)
            os.system('unzip ' + dataset + '/' + file + ' -d ' + dataset)
            os.system('rm ' + dataset + '/' + file)

    print("Linking files...")

    target_folder = '../7scenes_' + dataset + '/'

    def link_frames(split_file, variant):
        mkdir(target_folder + variant + '/rgb/')
        mkdir(target_folder + variant + '/poses/')
        mkdir(target_folder + variant + '/calibration/')

        with open(dataset + '/' + split_file, 'r') as f:
            split = f.readlines()	
        split = ['seq-' + s.strip()[8:].zfill(2) for s in split]

        for seq in split:
            files = os.listdir(dataset + '/' + seq)

            images = [f for f in files if f.endswith('color.png')]
            for img in images:
                os.system('ln -s ../../../'+src_folder+'/'+dataset+'/'+seq+'/'+img+ ' ' +target_folder+variant+'/rgb/'+seq+'-'+img)

            poses = [f for f in files if f.endswith('pose.txt')]
            for pose in poses:
                os.system('ln -s ../../../'+src_folder+'/'+dataset+'/'+seq+'/'+pose+ ' ' +target_folder+variant+'/poses/'+seq+'-'+pose)
            
            for i in range(len(images)):
                with open(target_folder+variant+'/calibration/%s-frame-%s.calibration.txt' % (seq, str(i).zfill(6)), 'w') as f:
                    f.write(str(focal_length))
    
    link_frames('TrainSplit.txt', 'training')
    link_frames('TestSplit.txt', 'test')
