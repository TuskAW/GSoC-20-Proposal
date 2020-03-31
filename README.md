# GSoC-20-Proposal
# Implementing Mesh RCNN in Tensorflow Graphics

## Contact Information

**Name :** Madhav Iyengar<br/>
**University :** [University of Southern California, Viterbi School of Computer Science](https://www.cs.usc.edu/)<br/>
**Email-id :** [thealmightylion.madhav@gmail.com](mailto:thealmightylion.madhav@gmail.com)<br/>
**GitHub username :** [MadhavEsDios](https://github.com/MadhavEsDios)<br/>
**LinkedIn :** [https://www.linkedin.com/in/madhav-iyengar-012398135/?originalSubdomain=in](https://www.linkedin.com/in/madhav-iyengar-012398135/?originalSubdomain=in)<br/>
**Time-zone :** GMT + 5:30

## Abstract

MeshRCNN is one of the network architectures which combines the best of both worlds of 2D perception and 3D shape understanding viz. it is able to predict fine-grained instance-wise 3D shapes within a scene by just looking at corresponding unconstrained real-life images (with multiple objects, occlusing, diverse lighting). The current implementation of MeshRCNN is in PyTorch, because of which Tensorflow developers are unable to take advantage of the proposed ideas. The objective of this project is threefold,
1. Extend Tensorflow-Datasets to include the Pix3D dataset which contains real-life 2D-3D pairs with pixel-level alignment 

2. Re-implement PyTorch3D modules like Cubify, Graph Convolutions, Vertex Alignment, Differntiable Mesh Sampling and Mesh Losses in Tensorflow-Graphics and 

3. Re-implement MeshRCNN in Tensorflow-Graphics.

This project will pave the way for Tensorflow based Computer Graphics/ Vision researchers to get the best out of the TF-Graphics module by providing them essential 3D functionalities / layers. It will also provide a basis for them to implement their own ideas within tf-graphics and will also allow deveopers to test new 2D-to-3D benchmarks on the MeshRCNN architecture. Apart from this, the Pix3D tensorflow-datasets Builder will provide a basis for other users to bring in their own 3D datasets into the tf-datasets API.
This project will also be a step towards the main goal of the Tensorflow Graphics project viz. creating a single fully differentiable machine learning system which combines computer vision (to extract scene parameters) and computer graphics (use scene parameters to render a 3D scene). 

### MaskRCNN:

MaskRCNN [\[1\]](https://arxiv.org/pdf/1703.06870.pdf)is a state-of-the-art instance segmentation / object detection pipeline proposed by He et al.
MaskRCNN builds upon the FasterRCNN architecture and introduces a Mask prediction branch as can be seen from this image:
![](image/)

However the success of MaskRCNN is majorly attributed to the ROI Align module which replaces the traditional ROI Pooling used in FasterRCNN. This is because RoI boundaries are quantized in RoI Pooling which leads to misalignment between the RoI and the extracted features. Although this does not impact classification as CNN’s are robust to minor translations, it heavily impacts mask/ object bounding box predictions. RoI Align solves this problem by not quantizing any boundaries i.e. preserving floating point coordinates and then uses bi-linear interpolation to accurately calculate the value of input features.


### MeshRCNN:

#### Motivation:

There have been rapid advances in the field of 2D perception tasks namely: object recognition, object localization, instance segmentation, and 2D keypoint prediction. These methods achieve impressive performance in their respective tasks despite being trained on cluttered real-life images. Despite their performance, these methods ignore one essential fact viz. the world around us and the objects within it are in 3D and not the XY image plane.

At the same time, significant advances have also been made in the field of 3D shape understanding using deep learning. These advances include a variety of novel architectures which can process different 3D shape representations such as voxels, point clouds, and meshes. However, most of these methods have been trained on synthetic datasets composed of rendered objects in isolation which are drastically less complex than 2D natural image benchmarks and real-life 3D objects.

Thus there is a need to develop systems which operate on unconstrained real-life images with many objects, occlusion and diverse lighting conditions (like previous work on 2D perception) but do not ignore the rich 3D structure of the world around us (like previous work on 3D shape prediction trained on synthetic benchmarks).

In an effort to work towards this goal, Gkioxari et al. propose MeshRCNN [\[2\]](https://arxiv.org/pdf/1906.02739.pdf) which jointly performs the tasks of 2D object detection and 3D shape prediction by using a fully differentiable end-to-end architecture. 

MeshRCNN builds upon the state-of-the-art 2D recognition architecture of MaskRCNN and introduces an additional ‘mesh prediction branch’ which outputs high-resolution triangle meshes and takes only RGB images as input. Thus, the authors of MeshRCNN have created an architecture that can predict accurate 3D shapes of multiple objects present within a real-life scene by just looking at the corresponding RGB image.

MeshRCNN builds upon the MaskRCNN architecture by introducing 2 new branches viz. the Voxel Branch and the Mesh Refinement Branch.

#### Voxel Branch:
The voxel branch accepts features from the MaskRCNN RoI Align module as input and is responsible for estimating a coarse voxelization of each object present within a scene and ultimately converting this into a coarse 3D mesh.
This branch consists of 3 main components:
1. **Voxel Occupancy Predictor**:
This is analogous to the Mask predictor branch of MaskRCNN viz. instead of predicting a mask on a 2D image this sub-module predicts voxel occupancies on a 3D grid which represents the full shape of the object.

2. **Cubify**:
This sub-module is responsible for creating a mesh from the coarse voxelized representation so that more fine-grained shapes can be predicted. The cubify sub-module does this by replacing each predicted occupied voxel with a cuboid triangle mesh with 8 vertices, 18 edges and 12 faces. Other optimizations like merging shared interior faces are also done to avoid redundancy.
The result of this module is a 3D mesh which has the same topology as the predicted voxels.

3. **Voxel Loss**:
This loss is defined as a binary cross-entropy loss between the predicted voxel occupancies and ground-truth voxel occupancies. This loss enables the network to learn more accurate voxel representation of objects which in turn will lead to better mesh prediction.

#### Mesh Refinement Branch:
The cubified branch obtained from the voxel branch provides an extremely coarse 3D shape and cannot accurately model fine-grained structures like legs of chairs, lamp posts etc.
The mesh refinement branch is thus responsible for refining the vertices of the coarse mesh obtained from the voxel branch to create a fine-grained mesh via a sequence of refinement stages. Each refinement stage consists of 3 operations:

1. **Vertex Alignment**:
This sub-module is responsible for generating image-aligned feature vectors for each mesh vertex. This is done by using the intrinsic matrix of the camera to project each vertex onto the image plane and then using bi-linear interpolation to calculate the image feature vector for each projected vertex. The ideas applied here are similar to those proposed by the authors of the Spatial Transformers paper[\[3\]](https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf).

2. **Graph Convolution**:
This sub-module uses several graph convolution layers (GCN) to aggregate information over local mesh regions. More specifically, the GCN layers are used to propagate the image-aligned features for each vertex obtained from Vertex Alignment along the mesh edges so that information can be aggregated. The idea used here is inspired by the GCN paper [\[4\]](https://arxiv.org/pdf/1609.02907.pdf) by Kipf et al.

3. **Vertex Refinement**:
To predict better and more fine-grained meshes it is essential to find the optimum arrangement of vertices. This is exactly why the Vertex Refinement sub-module is responsible for refining the mesh prediction by updating the mesh geometry while keeping the topology fixed. This refinement procedure will predict a semi-refined mesh which will be further refined by consequent stages. More specifically, a simple model of a learnable weight matrix combined with the tanh activation function is used to update the vertices.

The last component of the MeshRCNN architecture is the Mesh Loss.

#### Mesh Loss:
Defining losses which directly operate on triangular meshes is extremely challenging. This is why the authors of MeshRCNN propose to calculate the loss over a finite set of points instead of meshes. They achieve this by employing a differentiable mesh sampling technique which densely samples the surface of the mesh to give rise to a point cloud. This sampling is done with both the predicted and ground-truth meshes. Ultimately a weighted sum of the chamfer distance, normal distance and shape regularizing edge loss is used to force the network to predict smoother and more refined meshes.

### Pix3D Dataset:
We study 3D shape modeling from a single image and make contributions to it in three aspects. Pix3D is a large-scale benchmark of diverse image-shape pairs with pixel-level 2D-3D alignment. Pix3D has wide applications in shape-related tasks including reconstruction, retrieval, viewpoint estimation, etc. The Pix3D dataset solves 3 major problems with previous 3D shape benchmarks:
1. These datasets either contain only synthetic data **OR**
2. Lack precise alignment between 2D images and 3D shapes **OR**
3. Only have a small number of images
Apart from the above, another major contribution of the Pix3D dataset is that it proposes new evaluation criteria tailored specifically for the task of 3D shape reconstruction.

Given the objective of the authors of MeshRCNN to create an architecture which can train on real-life images and estimate 3D shapes of multiple objects within a scene, the Pix3D dataset turns out to be a perfect fit. This is because, not only does the Pix3D dataset consist of unconstrained real-life images but also has perfectly aligned 2D image-3D shape pairs.
The MeshRCNN network architecture which is customised for the Pix3D dataset is specified in Table No.9 in the original paper.
[]!(link)

### Tensorflow Datasets:
Open source datasets are the utmost essentials for making progress in machine learning research. Even though plenty of such datasets such as MNIST, Cycle GAN, Pix3D, etc. are publicly available, it is still extremely difficult to include them in a machine learning pipeline. The difficulties include writing separate scripts to download these datasets and difficulties with pre-processing them into a common format given their different formats & complexities.
The Tensorflow datasets module addresses these problems by performing the heavy-lifting of downloading and pre-processing datasets into a common format on disk. It does this by making use of the tf.data API and makes datasets easy to use by providing the functionality of accessing them as Numpy arrays.

Tensorflow Datasets provides the ability to easily integrate popular public datasets like MNIST, SVHN, Large Movies Dataset and 26 more. More importantly for our use case, tensorflow-datasets provides the option to add datasets via the Dataset Builders python class.
The Dataset Builders class has 3 methods which need to be implemented:
1. _info: This method contains all information about the features of the dataset.
2. _split_generators: This method is responsible for downloading and splitting the data into specified train / test splits.
3. _generate_examples: This method is used to populate feature placeholders specified in _info with actual instance feature values.

Extending Tensorflow Datasets by implementing Dataset builders for Pix3D:
To succeed in the goal of creating MeshRCNN in Tensorflow Graphics, it is necessary to create an input pipeline which can provide training / test/ validation data to the MeshRCNN architecture.
It is for this reason that I used the "Add a Dataset" feature provided in the tfds documentation and implemented the Dataset Builders class for the Pix3D dataset [Code](https://github.com/MadhavEsDios/tfds-pix3d). The code for the pix3d.py script that I have written is elucidated below:

```python
class pix3d(tfds.core.GeneratorBasedBuilder):
  """Short description of my dataset."""

  VERSION = tfds.core.Version('0.1.0')
  MANUAL_DOWNLOAD_INSTRUCTIONS = 'Testing'

  def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            # This is the description that will appear on the datasets page.
            description=("This is the dataset for Pix3D. It contains yyy. The "
                         "images are kept at their original dimensions."),
            # tfds.features.FeatureConnectors
            features=tfds.features.FeaturesDict({
                    'image': tfds.features.Image(),
                    'width': tfds.features.Tensor(shape=(1,), dtype=tf.uint8),
                    'height': tfds.features.Tensor(shape=(1,), dtype=tf.uint8),
                    '2d_keypoints': tfds.features.Tensor(shape=(None, None, 2), dtype=tf.float32),
                    'mask': tfds.features.Image(),
                    '3d_keypoints': tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
                    'voxel': tfds.features.Tensor(shape = (None, None, None), dtype=tf.uint8),
                    'rot_mat': tfds.features.Tensor(shape = (3, 3), dtype=tf.float32),
                    'trans_mat': tfds.features.Tensor(shape = (1, 3), dtype=tf.float32),
                    'focal_length': tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                    'cam_position': tfds.features.Tensor(shape=(3, ), dtype=tf.float32),
                    'inplane_rotation': tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                    'bbox': tfds.features.BBoxFeature(),
                    'metadata': {
                            'category': tfds.features.Text(),
                            'img_source': tfds.features.Text(),
                            'model': tfds.features.Text(),
                            #'model_raw': tfds.features.Text(),
                            'model_source': tfds.features.Text()
                            #'truncated': tfds.features.Text(),
                            #'occluded': tfds.features.Text(),
                            #'slightly_occluded': tfds.features.Text()
                            }
            }),
       
            homepage="http://pix3d.csail.mit.edu/",
            # Bibtex citation for the dataset
            citation=r"""@article{my-awesome-dataset-2020,
                                  author = {Smith, John},"}""",
        )
    
  def _split_generators(self, dl_manager):
        # this is just a testing script, which is why I downloaded the dataset and  
        extracted_path = dl_manager.manual_dir#.manual_dir("/Users/Madhav/Experiments/meshrcnn/pix3d")
        #dl_manager.download_and_extract("http://pix3d.csail.mit.edu/data/pix3d.zip")
    
        # Specify the splits
        return [
                tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "extracted_path": extracted_path
                },
            )
        ]
    
    
  def _generate_examples(self, extracted_path):
      json_path = os.path.join(extracted_path, "pix3d.json")
      with open(json_path) as pix3d:
          pix3d_info = json.loads(pix3d.read())
      for pix in pix3d_info:
          image_id = pix['img'][4:-4]
          width, height = pix['img_size']
          normalized_bbox = np.asarray(pix['bbox'], dtype=np.float32) / np.array([width, height, width, height])
          ymin, xmin, ymax, xmax = normalized_bbox
          yield image_id, {
                  "image": os.path.join(extracted_path, pix['img']),
                  "width": np.atleast_1d(width).astype(np.uint8),
                  "height": np.atleast_1d(height).astype(np.uint8),
                  "2d_keypoints": np.asarray(pix['2d_keypoints'], dtype=np.float32),
                  "mask": os.path.join(extracted_path, pix['mask']),
                  "3d_keypoints": np.loadtxt(os.path.join(extracted_path, pix['3d_keypoints']), dtype=np.float32),
                  "voxel": scipy.io.loadmat(os.path.join(extracted_path, pix['voxel']))['voxel'],
                  "rot_mat": np.asarray(pix['rot_mat'], dtype=np.float32),
                  "trans_mat": np.asarray(pix['trans_mat'], dtype=np.float32)[np.newaxis],
                  "focal_length": np.atleast_1d(pix['focal_length']).astype(np.float32),
                  "cam_position": np.asarray(pix['cam_position'], dtype=np.float32),
                  "inplane_rotation": np.atleast_1d(pix['inplane_rotation']).astype(np.float32),
                  "bbox": tfds.features.BBox(ymin = ymin, xmin = xmin, ymax = ymax, xmax = xmax),
                  "metadata": {
                          "category": pix['category'],
                          "img_source": pix['img_source'],
                          "model": pix['model'],
                          #"model_raw": pix['model_raw'].tostring(),
                          "model_source": pix['model_source']
                          #"truncated": pix['truncated'],
                          #occluded": pix['occluded'],
                          #"slightly_occluded": pix['slightly_occluded']
                          }
                  }
        

pix3dbuilder = pix3d()
info = pix3dbuilder.info
#print(info)
pix3dbuilder.download_and_prepare(download_config=tfds.download.DownloadConfig(manual_dir="/Users/Madhav/Experiments/meshrcnn"))
pix3dbuilder.as_dataset()
```
I have faced several issues while testing out this script because of which I have opened an issue in the tensorflow-datesets repo [#1610](https://github.com/tensorflow/datasets/issues/1610). My issues are currently being resolved, the 'Too many files' error could be a potential RAM issue, which I hope will be solved by running the script in Colab.

### Tensorflow Graphics:
The main objective of Tensorflow Graphics is to provide differentiable graphics layers which can easily be added to existing neural network architectures.
Some of the important functionalities offered by TF-Graphics are:
1. **Transformations**:
Object transformations control the position and poses of objects in space. Tf-Graphics provides differentiable transformation layers which can predict the the rotation/ translation matrices of an observed object with respect to a coordinate system. This is especially useful in tasks involving robotics where precise position estimation is required.
2. **Modelling Cameras**:
Tf-graphics provides mechanisms to learn where the camera should be positioned to achieve the desired scale of an object within a scene.
3. **Materials**:
Materials are extremely important in object rendering as they determine how an object appears within a scene by defining how external light affects them. Tf-Graphics takes this into account and provides a mechanism to predict material properties.
4. **3D Convolution and Pooling**
With significant developments in 3D representations like meshes and point clouds (obtained from LIDAR), it is important to perform convolution over such 3D volumes. This is why Tf-Graphics provides 3D convolutional and pooling layers.
5. **Tensorboard 3D**
Visual debugging is the best way to determine the accuracy of 3D prediction / rendering models. Tf-Graphics extends the popular Tensorboard to 3D and allows users to interactively visualize meshes and point clouds.

The main goal of Tensorflow Graphics is to provide users the tools to combine complex computer vision models with computer graphics models and create a differentiable pipeline which can render scenes by just looking at images.

This is exactly why a network architecture like MeshRCNN which simply takes images as input and gives accurate object-wise 3D meshes as output would be fulfilling the main tenet of Tensorflow Graphics. 

## Related Work

The authors of the MeshRCNN paper have published an official Github [Repository](https://github.com/facebookresearch/meshrcnn) written in PyTorch which is based on 2 PyTorch libraries viz. [Detectron2](https://github.com/facebookresearch/detectron2) and [PyTorch3D](https://github.com/facebookresearch/pytorch3d). Detectron2 is a library which allows the user to use /extend current state-of-the-art objects detectors and also provides pre-trained models of a varitey of network archtitectures. PyTorch3D is a module for 3D computer vision researchers written in PyTorch which provides data structures for storing and manipulating triangle meshes, efficient operations on triangle meshes (projective transformations, graph convolution, sampling, loss functions) and a differentiable mesh renderer among many other features. 

The main goal of my project is to imitate and re-implement these functionalities as a part of Tensorflow-Graphics. 


## Doubts which need clarification from mentors

**Base MaskRCNN model to be used**
- From my investigations up to now, I have observed that Detectron2 is Facebook Research’s version of the Tensorflow Object Detection API. Detectron2 provides the option of registering new architectures which allows users to override the behaviour of certain internal components of standard models including MaskRCNN [\[5\]](https://detectron2.readthedocs.io/tutorials/write-models.html). The PyTorch implementation of MeshRCNN takes advantage of this feature and adds the voxel and mesh refinement branches to the RoI Heads module to effectively create a trainable MeshRCNN architecture[\[6\]](https://github.com/facebookresearch/meshrcnn/blob/948bdcef6624ea3ac5cc1595e834416d95aec37f/meshrcnn/modeling/roi_heads/roi_heads.py). For the moment I see 2 potential ways to succeed in this project,
   1. Similar to Detectron2 I can use the ‘Create your own model’ option provided by TF Object Detection API [\[7\]](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/defining_your_own_model.md) and find a way to extend / modify the RoI Heads (RoI Align) module by adding the voxel and mesh refinement branches to effectively create a MeshRCNN architecture which can be trained by Tensorflow Object Detection API.
   2. Use the open-source implementation of Mask-RCNN by Matterport [\[8\]](https://github.com/matterport/Mask_RCNN) and modify it to create the MeshRCNN model.

**Should PyTorch3D operations used in MeshRCNN be submitted as individual PR's to tf-graphics**
- MeshRCNN uses several representations like Meshes / Voxels and essential operations like Cubify, Graph Convolutions, Sampling points from Meshes, Vertex Alignment, Subdividing Meshes and Mesh Edge Loss among many others from the PyTorch3D module. The former 3D representations are currently available in Tensorflow-Graphics however, I am not sure whether the latter operations are available. If not, I will consult with mentors / repo maintainers about re-implementing the above mentioned operations within tf-graphics and submit individual pull-requests for each operation. All of these operations as previously mentioned are part of PyTorch3D which is why I assume they should also be a part of Tensorflow Graphics. Another added benefit of submitting individual pull-requests for these operations would be that less time would be spent by both me and the mentors during the code-review period, as most of the code for this project would already have been reviewed / merged.

**Training Mechanism**
- I do not have a good GPU (GTX 660M) because of which I will be unable to train relatively large networks like MaskRCNN or an even bigger MeshRCNN. I wanted to know if it is OK if I carry out this project on Google Colab ? If not, will I be given remote access to a GPU/TPU cluster ? I have written the current proposal assuming that I will be training my model on Colab.

**Tensorflow Version**
- Which tensorflow version should this project be implemented in ? For this proposal, I assume it is TF 1.14

## Schedule of Deliverables

**Note1 :** I have no professional commitments till the beginning of my Masters in the end of August / early September. So I will be able to start coding 2 months earlier than the GSoC coding period, effectively giving me 5 months to complete my project. I would therefore be able to comfortably devote 50 hours a week for 5 months to complete this proposed project.

**Note2 :** In the possible eventuality that I am able to complete this project and get my pull requests merged before the deadline, I would also like to work towards implementing Occupancy Networks by Dr. Geiger’s group at Max Planck. If someone has already been selected for this project, I am open to taking project suggestions from my mentors.

**Note3 :** I have accounnted plenty of time to test each component written. This would assist me to spot and rectify bugs in my code quickly and also help me be confident about the correctness of the code that I write.

**Note4 :** Tensorflow-Graphics provides Tensorboard 3D to visualize meshes. I will definitely investigate this and identify whether it will help me visualize results from the intermediate / final stages of the MeshRCNN pipeline. If not, I will write a separate script which uses a renderer for eg. Blender.
Also, I will be making sure to monitor results on Tensorboard and keep checking if gradients are back-propagated correctly. I have accounted plenty of time for this task as well.

### March 31st - May 3rd, **Proposal Review Period**
- Even though I have lots of experience with using tensorflow and the tf.data API, I have never directly worked with the tensorflow-datasets and tensorflow-graphics modules. Since these are the 2 main modules on which this project is based, I intend to utilize this period of more than a month to investigate the tensorflow-datasets and tensorflow-graphics codebase. I have already started getting familiar with the tensorflow-datasets codebase as can be verified from my 
**PR[#1672](https://github.com/tensorflow/datasets/pull/1672)**,
which has been accepted and merged into the master branch. However, apart from this I also intend to implement small exercises involving both tf-datasets and tf-graphics on both my system and Colab. I am confident that my previous knowledge of the tf.data API and 3D computer vision will make this task easy.

- I also intend to resolve existing issues in these modules and submit PR’s to not only help further my knowledge but also give back to the open-source machine learning community.

- The official implementation of MeshRCNN by Facebook-Research is in PyTorch and is based on 2 libraries viz. Pytorch3D and Detectron2.  I have extensively worked with PyTorch while performing experiments for my [publication](https://pdfs.semanticscholar.org/3a7d/11af2b3833ec0f93aefc8b1f11a6fe271457.pdf). However, I have not yet worked with the aforementioned 2 libraries. Since the main objective of this project is to re-implement MeshRCNN in Tensorflow, I believe it is extremely important to first understand the existing implementation thoroughly. This is why I intend to investigate both Pytorch3D and Detectron2, and in the process also determine why these 2 libraries are being used in the official implementation. Again, I am confident that my experience in Pytorch and 3D computer vision will help me in finishing this task quickly.

- Complete an extensive literature review of the ideas used in the MeshRCNN paper. These include Cubify (A multi-threaded method which is proposed as an alternative to the MarchingCubes algorithm for converting coarse voxel representation into a mesh), Vertex Alignment (Similar to Spatial Transformers by Jaderberg et al.), Graph Convolutions (Similar to Semi-supervised classification with GCN’s by Kipf et.al at ICLR’17) and Mesh Losses (Given how calculating losses on meshes is not trivial, the authors use differentiable mesh sampling on the input mesh to effectively calculate the loss over a finite set of points i.e. a dense point cloud. This is similar to the idea presented in the Pix2Mesh paper by Wang et al.)

## Community Bonding Period

Before the official time period begins I intend to complete the first sub-task of this project i.e. extending tensorflow-datasets to include the Pix3D dataset.

### May 4th - May 8th
- Resolve issues with current implementation of pix3d.py
- Use tf.io operations wherever possible instead of native python io modules.
- Test script on both personal system and Colab to identify whether there are any issues with RAM. Fo eg. ‘too many files’ issue might be a RAM problem or due to incorrect closing of files in the script.

### May 11th - May 15th
- Analyse different features of the Pix3D dataset and their corresponding data formats.
- Create fake examples directory with a few fake examples which mimic Pix3D data by modifying the example script: [\[9\]](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/testing/fake_data_generation/cifar.py) or creating my own.
- In the process also ensure that fake example directory size is as small as possible.

 ### May 18th - May 22nd
- Implement unit testing script which builds upon the tfds.testing.DatasetBuilderTestCase class and uses the generated fake examples to verify if pix3d.py is indeed working as intended.
- Identify if all TODO() marked directories have been modified.
- Add import, URL checksums, citations and adjust coding style.
- Submit pull request to tensorflow-datasets repository.

 ### May 18th - May 22nd
- Address potential issues with Pix3D pull request.
- Finalise the chosen MaskRCNN model i.e. either TF Object Detection API or open-source Matterport implementation and begin analysing the ROI Heads module.
- Modify the parameters of Box and Mask branches so that they can make predictions on Pix3D images. 
- Verify whether modified model is able to train on Colab.
- Review Voxel Branch in current PyTorch implementation which includes predicting voxel occupancy, cubify and voxel loss.

 ### May 25th - May 29th, Begin Voxel Branch
- Identify how ROI Heads can be extended / modified to add the Voxel branch.
- Start implementing the voxel occupancy predictor which uses the camera intrinsic matrix to make voxel predictions aligned to the image plane.
- Begin implementing Cubify defined in PyTorch 3D [\[10\]](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/cubify.html#cubify). The current implementation uses simple PyTorch operations which are all available in Tensorflow, so re-implementing it should be straightforward. The authors of MeshRCNN clearly specify that they run Cubify as a batched operation. As can be seen in the appendix section they perform this by replacing loops with 3D convolutions. Thus there are no worries about re-implementing complicated multi-threaded operations.

## Coding Period Begins

 ### June 1st - June 5th
- Finish voxel occupancy predictor part of the Voxel Branch.
- Finish implementing Cubify method in tensorflow.
- Start implementing voxel occupancy prediction loss (binary cross-entropy loss) by re-implementing the voxel_rcnn_loss() method[\[11\]](https://github.com/facebookresearch/meshrcnn/blob/89b59e6df2eb09b8798eae16e204f75bb8dc92a7/meshrcnn/modeling/roi_heads/mesh_head.py).

 ### June 8th - June 12th
- Put the voxel branch together by combining the voxel predictor, cubify and voxel loss modules.
- Test if MaskRCNN + Voxel Branch is able to train by just adding the voxel loss to the MaskRCNN loss. (Check whether any shape mismatch errors occur and also verify whether gradients are being back-propagated accurately)
- Resolve potential issues with the voxel branch

 ### June 15th - June 19th, Begin Mesh Refinement Stage
- Debug and investigate how Vertex Alignment is implemented in PyTorch3D
- Begin implementation of Vertex Alignment either as a separate module or as a part of tf-graphics. There are 2 options for this viz. re-implement VertAlign by imitating the PyTorch3D implementation[\[12\]](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/vert_align.html#vert_align) OR utilize the official implementation of the Pixel2Mesh paper which is in tensorflow[\[13\]](https://github.com/nywang16/Pixel2Mesh). 

 ### June 22nd - June 26th
- Finish implementing the Vertex Alignment module.
- Perform unit tests and check whether the VertAlign module is able to process the output of the voxel branch as intended.

 ### June 22nd - June 26th
- Investigate how the GraphConv module is implemented in PyTorch3D.
- Begin re-implementation of Graph Convolution module which defines both forward and backward passes of a single graph convolution layer. There are 2 options here as well viz. re-implement the GraphConv by imitating the PyTorch3D implementation[\[14\]](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/graph_conv.html#GraphConv.__init__) OR directly use the official implementation of the Kipf et. al. GCN paper in tensorflow 1.12 [\[15\]](https://github.com/tkipf/gcn).

 ### June 29th - July 3rd
- Continue implementation of GraphConv module.
- Download and pre-process graph convolution benchmarks like Citeseer/ Cora/ Pubmed so that they can be used to test the implementation.

 ### July 6th - July 10th
- Finish implementation of GraphConv module.
- Independently train the module on either Citeseer/ Cora/ Pubmed and verify if GraphConv works as intended.
- Test whether GraphConv is able to work with the output from the VertAlign module.

 ### July 13th - July 17th
- Begin implementation of SubdivideMeshes module which subdivides a triangle mesh by adding a new vertex at the center of each edge and dividing each face into four new faces.
- For this Class, I will be imitating the current implementation provided in PyTorch3D [\[16\]](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/subdivide_meshes.html#SubdivideMeshes).

 ### July 20th - July 24th
- Finish implementing SubdivideMeshes module.
- Begin and finish investigation / implementation of Vertex Refinement function which involves using the tanh activation and a single learnable weight matrix.[\[17\]](https://github.com/facebookresearch/meshrcnn/blob/89b59e6df2eb09b8798eae16e204f75bb8dc92a7/shapenet/modeling/heads/mesh_head.py)
- Investigate PyTorch3D implementations of mesh_edge_loss[\[18\]](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_edge_loss.html#mesh_edge_loss), chamfer distance[\[19\]](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/chamfer.html#chamfer_distance) and differentiable sampling of points from meshes[\[20\]](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/sample_points_from_meshes.html#sample_points_from_meshes) which will be used to calculate the MeshRefinementStage Loss.

 ### July 27th - July 31st
- Begin implementation of mesh_edge_loss, chamfer distance and sample_points_from_meshes methods.

 ### Aug 3rd - Aug 7th
- Finish implementation of MeshRefinementStage Loss components. 
- Combine all 3 previously implemented classes viz. VertAlign, GraphConv and SubdivideMeshes along with the Vertex Refinement operation and the MeshRefinementStage Loss to create the MeshRefinementStage module.
- Create the Mesh Refinement Branch which for the Pix3D dataset version of MeshRCNN is 3 serialized instances of the MeshRefinementStage module.
- Add the Mesh Refinement branch to the existing pipeline (MaskRCNN + Voxel branch) to finish the MeshRCNN architecture.

 ### Aug 10th - Aug 14th
- Implement training and evaluation scripts which use evaluation metrics used in the paper.
- Begin writing script which will help visualize both intermediate (voxel branch, mesh refinement stages 1,2), the final mesh output and also the 2D object detections and masks.
- Begin training MeshRCNN model on Colab and keep track of gradients on TensorBoard to verify if gradients are being back-propagated correctly.
- Address potential training issues.

 ### Aug 17th - Aug 21st
- Continue addressing potential training issues.
- Complete training of model, verify if results match those published in the paper.
- Finish documentation of code written along with comments and style checks.
- Submit Pull Request consisting of Colab Notebooks to TF-Graphics repo.
- If time permits, I will also write a blogpost explaining what I have done.

 ### Aug 24th - Aug 28th
- Address any potential issues with either training or evaluation.
- Get feedback from mentors / repo maintainers and make changes.

 ### Aug 28th - Aug 31st
- Get Pull Request merged.

**Note :** I understand that my proposed time-schedule seems a bit stretched out and certain modules might have been allotted more time than required. This is because, I feel that this project might not be as straight-forward as it seems. However, I will do my best to finish this project before the beginning of August. If this is the eventuality that plays out, I would like to work on any other project before the coding period ends (preferably implementing Occupancy Networks).

## Future Work

I believe that implementing a poweful network like MeshRCNN which uses images to automatically predict accurate 3D shapes in perhaps the most popular machine learning language i.e. Tensorflow will allow developers to test their own benchmarks containing 2D-3D object pairs with ease paving the way for significant developments / research. 
My direction of future work would be to somehow adapt MeshRCNN to accept stereo-pairs, as such a model would effectively replicate our eyes (which also predict an accurate 3D model per object with a stereo-image setup). 

Apart from this, I plan to stick around and help others to resolve any issues related to this project which may emerge in the future.

## Development/ Research Experience

In my junior year, I worked on understanding recommender systems as part of a design project and published an oral [paper](https://ieeexplore.ieee.org/abstract/document/7934910/) titled “A collaborative filtering based model for recommending graduate schools” at the IEEE International ICMSAO’17 conference. In this work, I apply recommender systems to a personally motivating task of finding graduate schools and discuss the effects of various distance metrics on the quality of recommendations.

I worked as a Software Development Intern at ESRI (Environmental Systems Research Institute) for a period of 6 months in my final year of undergraduation. During my internship I have done many projects on different coding languages including developing geo-location mapping based workflows for NASA and Digital Globe(HYCOM, SST, GBDX, etc.) - in python
and developed an Add-In for the ArcGIS Pro enterprise software - in C# (which is the first user experience Add-In of ArcGIS Pro that allows users to access / open satellite data and pre-apply raster functions).

After graduation, I worked as a research assitant in the [Prof. Horst Bischof's](https://scholar.google.co.in/citations?user=_pq05Q4AAAAJ&hl=en) group [LRS](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/research/)(Learning-Recognition-Surveillance) group which is a part of [ICG](https://www.tugraz.at/institutes/icg/home/)(institute of Computer Graphics and Vision) at TU Graz, Austria (Graz University of Technology) for more than a 1.5 years.
During my time at LRS, most of my major contributions were made to the Austrian Government funded FFG projects of Dynamic Ground Truth (DGT) - 3D object detection and Static Ground Truth (SGT) - develop autonomous driving simulator based on real data. As part of the DGT project, I was able to publish an oral [paper](https://pdfs.semanticscholar.org/3a7d/11af2b3833ec0f93aefc8b1f11a6fe271457.pdf) “Detecting Out of Distribution Traffic Signs” published at the OAGM Workshop’19. In this work, I show that Out-of-Distribution(OOD) detection techniques like temperature distillation & adversarial loss based softmax calibration, etc. can be used to detect unseen categories of traffic signs i.e. signs which have never been encountered before by an autonomous vehicel. Here, I also show that non-Deep Learning techniques like the One-Class SVM and Linear SVM can be used as a strong baseline for Out-of-Distribution Detection.
Apart from these state-funded projects, I also made major contributions to the anonymization project, where faces of pedestrians/ vehicel license plates had to be blurred out in both image & video datasets to comply with EU privacy laws. This project is currently being used by companies like Audi, Siemens and Vexcel Imaging among many others.

For other experiences and more explanations, please refer to my resume.

Both these experiences have helped me gain priceless insight about what really goes into developing products for people on a large scale.

## Why this project?

Although, I have lots of research experience in Computer Vision and Machine Learning, I have never explicitly worked in the field of Computer Graphics. Working on this project will not only help me gain more insight about Computer Graphics, but also help me understand the nuances of 3D representations like meshes & voxels apart from helping me understand how 3D renderers work. Since, I will be starting my Masters this Fall, this project will also help me be in optimum coding condition before my program starts and help me gain confidence about implementing large-scale computer vision research projects of my own. While gaining so much knowledge, I also get to give back to the open-source machine learning community.
I also believe that implementing MeshRCNN in Tensorflow would pave the way for Tensorflow developers to not only get the best out of Tensorflow-Graphics but also help them implement their own ideas and test new benchmarks.

Hence, seeing the potential impact of the project and its overlap with my interest and previous work, this particular project piqued my interest.

## Appendix

Below is a list of all the web articles, libraries, code snippets and research papers mentioned in the proposal.

1 - https://arxiv.org/pdf/1703.06870.pdf<br/>
2 - https://arxiv.org/pdf/1906.02739.pdf<br/>
3 - https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf<br/>
4 - https://arxiv.org/pdf/1609.02907.pdf<br/>
5 - https://detectron2.readthedocs.io/tutorials/write-models.html<br/>
6 - https://github.com/facebookresearch/meshrcnn/blob/948bdcef6624ea3ac5cc1595e834416d95aec37f/meshrcnn/modeling/roi_heads/roi_heads.py<br/>
7 - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/defining_your_own_model.md<br/>
8 - https://github.com/matterport/Mask_RCNN<br/>
9 - https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/testing/fake_data_generation/cifar.py<br/>
10 - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/cubify.html#cubify<br/>
11 - https://github.com/facebookresearch/meshrcnn/blob/89b59e6df2eb09b8798eae16e204f75bb8dc92a7/meshrcnn/modeling/roi_heads/mesh_head.py<br/>
12 - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/vert_align.html#vert_align<br/>
13 - https://github.com/nywang16/Pixel2Mesh<br/>
14 - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/graph_conv.html#GraphConv.__init__<br/>
15 - https://github.com/tkipf/gcn<br/>
16 - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/subdivide_meshes.html#SubdivideMeshes<br/>
17 - https://github.com/facebookresearch/meshrcnn/blob/89b59e6df2eb09b8798eae16e204f75bb8dc92a7/shapenet/modeling/heads/mesh_head.py<br/>
18 - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_edge_loss.html#mesh_edge_loss<br/>
19 - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/chamfer.html#chamfer_distance<br/>
20 - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/sample_points_from_meshes.html#sample_points_from_meshes<br/>
