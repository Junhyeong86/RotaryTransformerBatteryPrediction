## Install
```
pip install "robat @ git+https://github.com/Junhyeong86/RotaryTransformerBatteryPrediction.git"
```

## Training yout own RoBaT model
### SOH Optimization
```
robat-train --multirun\
    training.training_args.num_train_steps=10000\
    training.training_args.learning_rate=5e-5,1e-5,5e-6\
    repeat=3
```
