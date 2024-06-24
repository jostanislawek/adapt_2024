import fastai
from fastai.vision.all import *
from PIL import Image
import timm
from torchvision import transforms
import paths as p

def getActivation(name):
  # the hook signature

  def hook(model, input, output):
    activation[name] = output.detach()

  return hook

# Define a hook function to capture the activations of the last layer
def hook_fn(module, input, output):
    print("Activation of head's last layer:")
    print(output)


def split_list(lst, chunk_size):
  return list(zip(*[iter(lst)] * chunk_size))


if __name__ == '__main__':
  # Load the data
  dls = ImageDataLoaders.from_folder(p.data_train_sample, train="Training", valid='Validation',
    item_tfms=Resize(224), batch_tfms=None, bs=4)

  # Load the finetuned model
  device = torch.device("cuda")
  model = timm.create_model('convnext_tiny', num_classes=5)
  #load_learner(p.model_path + 'learn1.pt')
  model.load_state_dict(torch.load(p.model_path + 'learn_dict1.pt'), strict=False)
  model.to(device) # Ensure that GPU is used
  model.eval()

  # Register forward hooks on the last layer
  module_list = [n for n, _ in model.named_modules()]

  # Print all the model layers
  # for i in module_list:
  # print(i)

  hook_handle = model.head.register_forward_hook(hook_fn)
  vector_list = []
  for cat in dls.train.vocab:
    #print(dls.items)
    print("CATEGORY")
    print(cat)
    for batch in dls.train:

      inputs, *other = batch
      out = model(inputs)
      out_c = out.detach().cpu().numpy()
      vector_list.append(out_c)

  # Get np vectors saved in the folder
  img_name_list = []
  for i in dls.items:
    i = str(i)
    try:
      image_name = i.rsplit("\\", 2)[-2:]
      img_name_list.append(image_name)

    except Exception as e:
      print(f"Error processing item ({i}): {e}")

# Print paths and image names
  for vector, img_path in zip(vector_list, img_name_list):

      print(" I AM THE IMAGE PATH")
      print(img_path)

      just_image_name = img_path[1]
      just_image_name = just_image_name[:-5]


      # Create the directory if it doesn't exist
      save_dir = os.path.join(p.vector_path_train, img_path[0])
      print(p.vector_path_train)
      print(img_path[0])
      print(save_dir)
      print(os.path.exists(save_dir))
      os.makedirs(save_dir, exist_ok=True)

      filename = os.path.join(p.vector_path_train, img_path[0], just_image_name + ".npy")
      print(filename)
      np.save(filename, vector)

  # Detach the hooks
  hook_handle.remove()