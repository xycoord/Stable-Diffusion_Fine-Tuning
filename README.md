# "Marigold" Transparency Segmentation v1 (SD2)

This is the code for part of my final year project on Transparent Object Image Segmentation using a Marigold inspired method.

Marigold has since published their training code [here](https://github.com/prs-eth/Marigold)

For an updated, neater, version of this project using Stable Diffusion 3 see [v2](https://github.com/xycoord/Transparency-Estimation)


#### Run the Inference of Monodepth estimation: 

```
cd scripts
sh inference.sh
``` 

#### Run the Inference of Monodepth Training, Using SceneFlow as an example:
```
cd scripts
sh train.sh
``` 

Note the training at least takes 21 VRAM even the batch size is set to 1.