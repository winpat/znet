# znet

A toy library for building neural networks. This was built solely for
educational purposes to learn the zig programming language and freshen up my
understanding of neural networks.

Here is an example of how training a model based on the
[iris](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
dataset looks like:

``` zig
var features, var labels = try iris.load(allocator, "data/iris.csv");
minMaxNormalize(f32, &features);

var net = Network(f32).init(allocator, 4, 3);
defer net.deinit();

try net.addLinear(8);
try net.addReLU();
try net.addLinear(3);
try net.addSoftmax();

try net.train(300, 0.01, 32, features, labels);
```

## Development

This project is developed with Nix. To get an development shell run:

```
$ nix develop
```

There is a `Justfile` for the `just` command runner, containing recipes for
building the code and running the tests:

```
$ just test
```

The repository does not contain any datasets, you can load them from the
internet by running:

```
$ just download-datasets
```

> You don't need to use Nix or just to build this project. As long as you have
> the Zig 0.12 compiler, you can easily copy the build steps from the Justfile
> and execute them manually.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[BSD 3](LICENSE)
