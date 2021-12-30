# RecapD

[Improving Text-to-Image Generation by Discriminator with Recaption Ability]()

<img src="framework.png" witdh="900px" height="250px"/>

### Installation

- Clone this repository
```
git clone https://github.com/POSTECH-IMLAB/RecapD.git
```
- Create conda enviroment and install all the dependencies
```
cd RecapD
conda create -n recapD python=3.6
conda activate recapD
pip install -r requirements.txt
```

### Preparation
1. Download the preprocessed metadata for [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to ```datasets/```
```
export PROJECT_DIR=~/RecapD # path for project dir
mkdir datasets
cd datasets
gdown https://drive.google.com/uc?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9
unzip coco.zip
cd $PROJECT_DIR
```
2. Download [coco](http://cocodataset.org/#download) dataset and extract the images and annotations to ```datasets/coco```
3. Download the [pre-trained text encoder](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ) for coco and save it to ```datasets/DAMSMencoders/coco```
```
cd $PROJECT_DIR/datasets
mkdir DAMSMencoders
cd DAMSMencoders
gdown https://drive.google.com/uc?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ
unzip coco.zip
cd $PROJECT_DIR
```
5. Build vocabulary for recaptioning model
  ```
  python scripts/build_vocabularay.py \
         --captions datasets/coco/annotations/captions_train2014.json \
         --vocab-size 10000 \
         --output-prefix datasets/vocab/coco14_10k \
         --do-lower-case
  ```


### Train RecapD

```
python scripts/train_recapD.py
```

### Test RecapD
Download [pretrained RecapD](https://drive.google.com/file/d/1or9fpMC6-cCVCGol39f_kOI1Vc0fZvBT/view?usp=sharing) and save it to ```exps/256_cond_cap/checkpoint.pth```
Note that the words in the text should be in the vocabulary of DAMSM text encoder 

- Generate an image from a sentence
```
python scripts/demo.py --text "a computer monitor next to a keyboard laptop and a mouse"
```

- Generate images from sentences

Add texts for generation in ```example_sentences.txt``` 
```
python scripts/gen_recapD.py
```
Generated samples

<img src="samples.png" witdh="900px" height="250px"/>

### Reference
The code is based on [DF-GAN](https://github.com/tobran/DF-GAN) and [VirTex](https://github.com/kdexd/virtex)
