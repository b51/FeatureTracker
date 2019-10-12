## FeatureTracker

### What this repository for
Detect, match and track different type features

**Current type of feture points**
  - [X] ORB feature points

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
