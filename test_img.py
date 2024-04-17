import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from test_options import TestOptions
from dataset import CreateDataLoader
from model import create_model
from PIL import Image

opt = TestOptions().parse()
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
save_dir = os.path.join("./result/")
print("dataset size:   ",len(dataset))

for i, data in enumerate(dataset):
    model.set_input(data)
    output = model.predict()

    img_path = model.get_image_paths()
    filename = 'img_{:03d}.jpg'.format(i+1)
    print('process image... %s' % img_path)
    image_pil = Image.fromarray(output)
    s_path = os.path.join(save_dir,filename)
    image_pil.save(s_path)