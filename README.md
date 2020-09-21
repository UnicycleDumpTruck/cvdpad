# cvdpad

## Computer Vision Direction Pad Game Controller

This is a rough hack, not really intended for widespread use. I've been jealous of Scratch's "Video Sensing", but I like the MakeCode Arcade platform better for introducing programming to kids. The cvdpad.py script measures the optical flow (motion direction) of the whole camera view, and translates it into WASD keypresses sent to whatever application has the operating system focus. It's a bit of a kludge in that regard, but this is really more about me having a starting point, and seeing what level of performance I could achieve. Once you get the hang of jumping, it's perfectly playable, and a fair workout. More like Crossfit Crab, amiright?

Be sure to experiment with the threshold commandline argument, such as running it "python cvdpad.py -th 3". I also disabled the "down" direction pretty early on, and it now gets weird and slow when I uncomment the "down" section, so I probably broke it messing around with other things. I created a couple dead bands to separate "up" from "left" and "right", because it seemed like "up" was too easy to trigger.
