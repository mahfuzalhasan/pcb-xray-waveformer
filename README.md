# Parameter Efficient Fine Tuning Segment Anything Model for PCB Inspection

The Segment Anything Model (SAM), recognized for its flexibility in semantic segmentation, has proven effective in segmenting a wide range of objects in natural images. However, it faces limitations when applied to the complex layouts of PCB X-ray images, struggling to accurately identify and segment the various components embedded in circuit designs. 
Addressing these challenges requires adapting SAM to handle the intricate structures and X-ray artifacts unique to PCBs, necessitating targeted modifications and optimization. In this work, we present a tailored approach for segmenting PCB components from X-ray images by using a version of SAM, enhanced through parameter-efficient fine-tuning and few-shot learning techniques. We propose specific adjustments to improve the model’s capacity to capture detailed spatial relationships and accurately segment individual components. 
Our approach emphasizes efficient adaptation of the SAM model to manage the unique features of PCB X-ray images.
Details of our our can be found at: (https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13138/3027646/Optimizing-the-segment-anything-model-for-PCB-component-segmentation-in/10.1117/12.3027646.full)
![model](https://github.com/user-attachments/assets/8a7a3952-5f97-431d-a6d7-f8f626e8fdcb)
Please cite our work:
@article{Roy2024OptimizingTS,
  title={Optimizing the segment anything model for PCB component segmentation in x-ray images through few-shot parameter-efficient fine-tuning},
  author={Antika Roy and Md Mahfuz Al Hasan and Shajib Ghosh and Nitin Varshney and Patrick J. Craig and Charles Woychik and Reza Forghani and Navid Asadizanjani},
  journal={Applications of Machine Learning 2024},
  year={2024},
  url={https://aphttps://doi.org/10.1117/12.3027646}
}
Our work is an extention and application on the following work from Meta.
# Segment Anything

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

![SAM design](assets/model_diagram.png?raw=true)
