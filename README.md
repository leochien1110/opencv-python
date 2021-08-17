Run dnn recognition with googlenet model

```
python3 deep_learning_with_opencv.py --image res/spaceshuttle.jpg --prototxt model/bvlc_googlenet.prototxt --model model/bvlc_googlenet.caffemodel --label model/synset_words.txt
```

List in resource directory: 

`b747.jpg`
`dogs.jpg`
`spaceshuttle.jpg`
`spaceshuttle_b747.jpg`

Run webcam dnn with opencv model
```
python3 detect_faces_video.py --prototxt model/deploy.prototxt.txt --model model/res10_300x300_ssd_iter_140000.caffemodel
```