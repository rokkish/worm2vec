# MNIST-rot experiments
This folder contains the script to run the MNIST-rot experiments.

#1 Run the model
The model is defined in `mnist_model.py`. To run the code, just run  
```bash
python run_mnist.py --combine_train_val True
```
This should download the MNIST-rot dataset and setup a log and checkpoint
folder, in which results are saved. The default settings for the model are
those we have arrived at for this task. Feel free to experiment with them. If
you find anything interesting, or any bugs for that matter, we'll be happy to
hear from you.

TODO
- [ ] Include pretrained model

## Fix bags
Experiments on python3.5, tensorflow(docker image from "tensorflow/tensorflow:1.9.0-gpu-py3" )

### Owing to python2 -> python3
* urlib2 -> urlib.request
* xrange -> range
* iteritems -> items

### Other
* values type not matched: (int, int, float) -> (int, int, int).
```python
    #Before
    new_shape = tf.concat(axis=0, values=[Ysh[:3],[max_order+1,2],[Ysh[3]/(2*(max_order+1))]])
    #After
    new_shape = tf.concat(axis=0, values=[Ysh[:3],[max_order+1,2],[int(Ysh[3]/(2*(max_order+1)))]])

```
* tf.nn.moments: ndarray -> list
```python
    #Before
    batch_mean, batch_var = tf.nn.moments(X, np.arange(len(Xsh)-3), name=name+'moments')
    #After
    batch_mean, batch_var = tf.nn.moments(X, list(range(len(Xsh)-3)), name=name+'moments')

```
* format error
```python
    #Before
    sys.stdout.write('{:d}/{:d}\r'.format(i, data['train_x'].shape[0]/args.batch_size))
    #After
    sys.stdout.write('{}/{}\r'.format(i, data['train_x'].shape[0]/args.batch_size))
```
## Other fixment

* limit GPU
```python
    config.gpu_options.allow_growth = False
    config.gpu_options.visible_device_list = "2"
    config.log_device_placement = True
```