# Show available recipes
default:
	just -l

# Download iris dataset
download-iris-dataset:
	curl https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv > data/iris.csv
	sed 's/"//g' -i data/iris.csv