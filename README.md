<!--
*** Template auto-generated
-->

<!-- PROJECT SHIELDS -->
<!-- START readme-templates/header.md -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
<!-- END readme-templates/header.md -->

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/TEAM-IMT/energy-network-optimization">
    <img src="/images/logo.png" alt="Logo">
  </a>

  <h2 align="center"><b> Energy Network Optimization </h2></b>

  <p align="center">
    District Heating Energy Network Optimization, based on ILS, PULP and MIP algorithms
    <br />
    <a href="https://github.com/TEAM-IMT/energy-network-optimization"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/TEAM-IMT/energy-network-optimization">View Demo</a>
    ·
    <a href="https://github.com/TEAM-IMT/energy-network-optimization/issues">Report Bug</a>
    ·
    <a href="https://github.com/TEAM-IMT/energy-network-optimization/issues">Request Feature</a>
  </p>
</p>

<!-- START readme-templates/2-table_contents.md -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>
<!-- END readme-templates/2-table_contents.md -->

<!-- ABOUT THE PROJECT -->
## About The Project  
In many countries around the world, the ability to heat and supply hot water to buildings is essential, currently, several studies are being carried out in order to determine the most efficient way to do it, one of these is through a Distric heating (DH). A DH is a system for distributing heat generated in a centralized location through a system of insulated pipes for residential and commercial heating requirements such as space heating and water heating in cold areas.

This repository seeks to buy the performance of PULP as an optimizer for a mixed integer linear programming (MIP) problem and an Integrated local serch (ILS) designed for a DH, for a specific energy network.


### Built With  
The main frameworks and libraries used in the project.
* [python](https://rasa.com/)
* [numpy](https://numpy.org)
* [scipy](https://www.scipy.org)
* [matplotlib](https://matplotlib.org/)
* [pandas](https://opencv.org/)
* [pulp](https://coin-or.github.io)

<!-- GETTING STARTED -->
## Getting Started  
To get a local copy just executed the following command:

```sh
git clone https://github.com/TEAM-IMT/energy-network-optimization
```

### Prerequisites  
Nothing to do

### Installation  
1. Install all the libraries

```sh
pip3 -m install -U -r requirements.txt
```

<!-- USAGE EXAMPLES -->
## Usage  
The program consists of three parts. For more information, please refer to the general/each part help.

```python
python3 main.py -h
python3 main.py PULP -h
python3 main.py ILS -h
python3 main.py ILSgraphics -h
```

Typical examples of program use:
```python
### PULP
python3 main.py PULP # Execute PULP algorithm
python3 main.py PULP -ILS results/output_33229254_greddy4_pert1_ls0_acep0.001.json # Execute PULP algorithm and comparated with best ILS solution 

### ILS
python3 main.py PULP # Execute ILS algorithm with default parameters. Save results in local directory

### ILSgraphics
python3 main.py ILSgraphics -dfp results/ -s localSearch.ini z_history.best -sp # Print all important results.
python3 main.py ILSgraphics -dfp results/best-worse-results/ -s z_history.acceptance acceptance_history -sp # Print results to best-worse solutions.
python3 main.py ILSgraphics -dfp results/output_34679923_random4_pert1_ls0_acep0.005.json # Print all results to best solution.
```

<!-- START readme-templates/6-roadmap.md -->
## Roadmap  
See the [open issues](https://github.com/TEAM-IMT/energy-network-optimization/issues) for a list of proposed features (and known issues).<!-- END readme-templates/6-roadmap.md -->

<!-- START readme-templates/7-contribution.md -->
## Contributing  
Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request<!-- END readme-templates/7-contribution.md -->

<!-- START readme-templates/8-license.md -->
## License  
Distributed under the MIT License. See [`LICENSE`](https://github.com/TEAM-IMT/energy-network-optimization/blob/main/LICENSE) for more information.<!-- END readme-templates/8-license.md -->

<!-- START readme-templates/9-contact.md -->
## Contact  
* **Johan Mejia** (johan-steven.mejia-mogollon@imt-atlantique.net) - [![Linkend][linkedin-shield]][linkedin-url-1]
* **Tatiana Moreno** (jenny-tatiana.moreno-perea@imt-atlantique.net) - [![Linkend][linkedin-shield]][linkedin-url-2]
* **Diego Carreño** (diego-andres.carreno-avila@imt-atlantique.net) - [![Linkend][linkedin-shield]][linkedin-url-3]  
**Project Link:**  [https://github.com/TEAM-IMT/energy-network-optimization](https://github.com/TEAM-IMT/energy-network-optimization)<!-- END readme-templates/9-contact.md -->

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements  
* [Best README Template](https://github.com/othneildrew/Best-README-Template)
* [Action dynamic readme](https://github.com/varunsridharan/action-dynamic-readme/)


<!-- MARKDOWNS AND LINKS -->
<!-- START readme-templates/links.md -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/TEAM-IMT/energy-network-optimization.svg?style=for-the-badge
[contributors-url]: https://github.com/TEAM-IMT/energy-network-optimization/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/TEAM-IMT/energy-network-optimization.svg?style=for-the-badge
[forks-url]: https://github.com/TEAM-IMT/energy-network-optimization/network/members
[stars-shield]: https://img.shields.io/github/stars/TEAM-IMT/energy-network-optimization.svg?style=for-the-badge
[stars-url]: https://github.com/TEAM-IMT/energy-network-optimization/stargazers
[issues-shield]: https://img.shields.io/github/issues/TEAM-IMT/energy-network-optimization.svg?style=for-the-badge
[issues-url]: https://github.com/TEAM-IMT/energy-network-optimization/issues
[license-shield]: https://img.shields.io/github/license/TEAM-IMT/energy-network-optimization.svg?style=for-the-badge
[license-url]: https://github.com/TEAM-IMT/energy-network-optimization/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url-1]: https://www.linkedin.com/in/johansmm/
[linkedin-url-2]: https://www.linkedin.com/in/tatiana-moreno-perea/
[linkedin-url-3]: https://www.linkedin.com/in/diego-andres-carre%C3%B1o-49b2ab157/
<!-- END readme-templates/links.md -->
