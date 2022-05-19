Code and data for the research article titled “Physics Informed LSTM Network for Flexibility Identification in Evaporative Cooling Systems”, authored by [Manu Lahariya](https://mlahariya.github.io/), Farzaneh Karami, [Chris Develder](http://users.atlantis.ugent.be/cdvelder/) and Guillaume Crevecoeur.

## Abstract
In energy intensive industrial systems, an evaporative cooling process may introduce operational flexibility. Such flexibility refers to a system’s ability to deviate from its scheduled energy consumption. Identifying the flexibility, and therefore, designing control that ensures efficient and reliable operation presents a great challenge due to the inherently complex dynam- ics of industrial systems. Recently, machine learning models have attracted attention for identifying flexibility, due to their ability to model complex nonlinear behavior. This research presents machine learning based methods that integrate system dynamics into the machine learning models (e.g., Neural Networks) for better adherence to physical constraints. We define and evaluate physics informed long-short term memory networks (PhyLSTM) and physics informed neural networks (PhyNN) for the identification of flexibility in the evaporative cooling process. These physics informed networks approximate the time-dependent relationship between control input and system response while enforcing the dynamics of the process in the neural network architecture. Our proposed PhyLSTM provides less than 2% system response estimation error, converges in less than half iterations compared to a baseline Neural Network (NN), and accurately estimates the defined flexibility metrics. We include a detailed analysis of the impact of training data size on the performance and optimization of our proposed models.

## Contents
This repository presents the codes for the proposed physics informed neural network variants along with the files used to simulate training and test data sets. The models proposed in the research article: (1) PhyLSTM, and (2) PhyNN.

## Citation
    @ARTICLE{LAHARIYA2022,
      author={Lahariya, Manu and Farzaneh, Karami and Develder, Chris and Crevecoeur, Guillaume},
      journal={IEEE Transactions on Industrial Informatics}, 
      title={Physics Informed LSTM Network for Flexibility Identification in Evaporative Cooling System}, 
      year={2022},
      volume={},
      number={},
      pages={1-1},
      doi={10.1109/TII.2022.3173897}
    }

## Acknowledgment
This research was performed at the [AI for Energy Research Group](https://ugentai4e.github.io/) at [IDLAB, UGent-imec](https://www.ugent.be/ea/idlab/en). Part of this research has received funding from the European Union's Horizon 2020 research and innovation programme for the projects [BRIGHT](https://www.brightproject.eu/), [RENergetic](https://www.renergetic.eu/) and [BIGG](https://www.bigg-project.eu/).

## Contact
If you have any questions, please contact me at manu.lahariya@ugent.be