run Index main to generate the necessary files to run a search
run Search main to execute a search and benchmark the outcomes vs the paper: https://arxiv.org/pdf/2405.12497

running: 
1. index DIRECTORY_TO_DATASET DATASET_NAME NUM_CENTROIDS DIMENSIONS
    * eg index "/Users/jwagster/Desktop/gist1m/gist/" gist 4096 4096
2. search DIRECTORY_TO_DATASET DATASET_NAME NUM_CENTROIDS DIMENSIONS B_QUERY OUTPUT_PATH
    * eg search "/Users/jwagster/Desktop/gist1m/gist/" gist 4096 4096 4 "/Users/jwagster/Desktop/gist1m/ivfrn_output/"

