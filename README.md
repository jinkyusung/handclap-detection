# **Hand-clap Detection Algorithm**

## **I.** Intro
To detect abnormal action of preschoolers during block building behavior, we implemented hand-clap detector based on a `mediapipe` landmarks metrics

## **II.** Architecture Details

### Step1. Detect Left, Right Hands
Just Use Mediapipe Hand-landmarker.

### Step2. Calculates some metrics
* L2-Distance
* Normal Vector of each palm surfaces
* Included Angles
* Direction with axis of $x, y, z$ of mediapipe coordinates.  

> If there is no detection of left or right hand, then put zero as value of each metrics.

### Step3. Find Outliers
By using moving average of distance, Find index of Outliers and  
_**Normalize**_ by other metrics. *(*Actually, **this** is removed in code since there is no performance increasing)*


## III. Environment Set-up

### Step1. Download this python script
Use `git clone` or directly download.
Then you can get a below hierarchy.
```
clap
├── data
│   └── data_for_loadmode.npy
├── videos
│   └── your_video.mp4
├── image
├── detect.py
├── main.py
├── metric.py
├── misc.py
├── README.md
└── requirements.txt
```


### Step2. Create anaconda virtual environment
```
(base) ~/clap> conda create -n env_name
(base) ~/clap> conda activate env_name
(env_name) ~/clap> conda install python=3.10.13
(env_name) ~/crop> pip install -r requirements.txt
```

### Step3. Run & test
```
(env_name) ~/clap> python main.py [-s/-l] ./videos/your_video.mp4
```
Note that `-l` is Load mode and `-s` is Save mode.
*or* Use Real-time mode with `-r`
```
(env_name) ~/clap> python main.py -r
```

## IV. Requirements
```
python=3.10.13
opencv-python=4.9.0.80
numpy=1.26.3
tqdm=4.66.2
mediapipe=0.10.9
```
