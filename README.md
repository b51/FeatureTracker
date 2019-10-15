## FeatureTracker

### What this repository for
Detect, match and track different type features

**Current type of feture points**
  - [X] ORB feature points
  - [X] SuperPoint feature points

### Usage
**Build**
```bash
git clone https://github.com/b51/FeatureTracker
cd FeatureTracker
mkdir build
cd build && cmake ..
make -j4
```

**Run**
```bash
./FeatureTracker --image_dir /path/to/images_dir [--image_suffix .jpg]
```

Kitti:

**ORB Tracker**

<img src="utils/orb_track.gif" width="1200">


**SuperPoint Tracker**

Resized image to 414, 125 (width/3, height/3)
<img src="utils/sp_track_resized_1_3rd.gif" width="1200">

Original size image
<img src="utils/sp_track_origin_size.gif" width="1200">
