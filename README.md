# Team-Vision
This Repo is dedicated to Team Vision regarding our group project for Step Presentation 2. <br/>
Create env

<b>1)</b> Start with Template.py were we create the folder structure and necessary files to support our project and pipelining
<br/> run  >python Template.py <br/>

<b>2)</b>python setup.py  Since our project is also a package ,and we have seen many library having such format <a href="https://pypi.org/project/seaborn/"> 
<br/>to Creating such structure we use setup.py. <br/>

<b>3)</b> for the metadata we use <b>setup.cfg</b> weher wee mentioned Licence ,OS,minimum python version,options where we need some third party dependency.
</br>

<b>4)</b>tox.ini since we have written the code for os independent and python version specified we write automatated test to handle this in tox.ini<br/>
where it create test environemnt for specified version and test all the specified task.automated test environment.

<b>(5</b> create the environment init_setup.sh</br> for the environment setup and run it using<br/> <h4> bash init_setup.sh</h4> <br/>