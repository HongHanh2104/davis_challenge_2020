from pathlib import Path
import glob
import os

def create_new_folder(root,
                      data_folder):
    new_folder = 'new_' + data_folder
    os.system('mkdir {}'.format(os.path.join(root, new_folder)))
    [os.system('mkdir -p {}/{}'.format(os.path.join(root, new_folder), x)) for x in ['JPEGImages', 'Annotaions']]
    return new_folder

def pascal_preprocess(root,
               data_folder,
               new_folder='new_pascal-5'):
    # check type folder
    
    data_path = os.path.join(root, new_folder)
    dst_img_path = os.path.join(data_path, 'JPEGImages')
    dst_ann_path = dst_img_path.replace('JPEGImages', 'Annotations')
    
    class_files = []
    for _file in os.listdir(os.path.join(root, data_folder)):        
        file_path = os.path.join(root, data_folder, _file)
        class_files.append([glob.glob('{}/*.txt'.format(os.path.join(file_path, x))) for x in ['train', 'test']])
    class_files = sum(class_files, [])
    
    for i in range(len(class_files)):
        for c in class_files[i]:
            dirname = os.path.dirname(c)
            namefile = os.path.basename(c).split('.')[0]
            if os.path.exists(os.path.join(dst_img_path, namefile)) == False:
                os.system('mkdir {}'.format(os.path.join(dst_img_path, namefile)))
            if os.path.exists(os.path.join(dst_ann_path, namefile)) == False:  
                os.system('mkdir {}'.format(os.path.join(dst_ann_path, namefile)))      
            
            with open(c, 'r') as f:
                print(dirname)
                for x in f.readlines():
                    #print(str(x) + '.jpg')
                    #print('cp {}/origin/{} {}'.format(dirname, x.strip() + '.jpg', os.path.join(dst_img_path, namefile)))
                    if 'train' in c:
                        os.system('cp {}/origin/{}.jpg {}'.format(dirname, x.strip(), os.path.join(dst_img_path, namefile)))
                        os.system('cp {}/groundtruth/{}.jpg {}'.format(dirname, x.strip(), os.path.join(dst_ann_path, namefile)))
                        #print("finish {} in {}.".format('train', x))
                    elif 'test' in c:
                        os.system('cp {}/origin/{}.jpg {}'.format(dirname, x.strip(), os.path.join(dst_img_path, namefile)))
                        os.system('cp {}/groundtruth/{}.jpg {}'.format(dirname, x.strip(), os.path.join(dst_ann_path, namefile)))
                        #print("finish {} in {}.".format('test', x))
        print("Finish {}".format(c))   
    print('Done')
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Path to dataset')
    parser.add_argument('--data_folder', help='The folder of dataset')
    args = parser.parse_args()

    create_new_folder(args.root, args.data_folder)
    pascal_preprocess(root=args.root, 
               data_folder=args.data_folder, 
               )




