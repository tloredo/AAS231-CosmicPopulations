# AAS231-CosmicPopulations

This repo contains content for a workshop on on hierarchical modeling of cosmic populations being held at the [AAS 231 meeting](https://aas.org/meetings/aas231), in Washington, DC, on 7 January 2018.  Topics include:

* Quick intro to Bayesian inference
* Hierarchical Bayes for estimating distributions and correlations with noisy data
* Handling selection effects with thinned latent point process models
* Posterior sampling and MCMC
* Approximate Bayesian Computation (ABC)

The workshop will include hands-on demonstrations using Python (in Jupyter notebooks), R, and [Stan](http://mc-stan.org/) via its [PyStan](http://mc-stan.org/users/interfaces/pystan.html) Python interface.  The presenters will be [Tom Loredo (Cornell astronomer)](http://www.astro.cornell.edu/staff/loredo/) and [Jessi Cisewski (Yale statistician)](http://statistics.yale.edu/people/jessi-cisewski).

## Computer preparation for the workshop

Please prepare you computer for the workshop **before the workshop**. There will not be time at the workshop for installing software, and the local WiFi resources are not intended for supporting the large downloads that may be required.

**Do not postpone this until the last minute.**  Depending on what software you may already have installed, some steps of this setup may require time-consuming downloads.

We will presume participants have basic familiarity with terminal-based command line computing, and with the Python language and Jupyter notebooks. Some familiarity with R may be helpful, but the R examples should be accessible to Python programmers.

To run the lab materials, you will need the following resources (version numbers are those we used for testing; earlier versions may suffice in some cases):

* Python 3.6
* IPython 6.1
* The PyData stack (NumPy, SciPy, matplotlib...)
* PyStan
* A C++ compiler compatible with Python (for PyStan)
* R and the RStudio IDE

If you already have these resources installed and are feeling adventurous, feel free to use your own installation. But please note that the limited time we have for the workshop won't allow us to help with problems associated with specific installations.

We *strongly* recommend that you use the `cospop18` Anaconda environment described below for running the lab materials. It's what we have used for developing the materials.  If you already have a different Python/R distribution and are worried about conflicts, consider the Miniconda option mentioned below—it will minimally alter your default command-line environment, and install the new material in an encapsulated manner.  If you are particularly concerned about conflicts, consider setting up a separate user account on your computer to use for the workshop (Anaconda installs material only in the user's account.)

We will be revising lab materials up to the time of the workshop; please postpone downloading and running the material until then.

### Software prerequisites for `cospop18`

**Windows users (others may skip this paragraph):** Neither presenter regularly uses the Windows OS, so unfortunately we can offer little advice specifically for Windows users.  Stan, which we will use heavily, requires a C++ compiler with libraries compatible with the user's Python installation. This is somewhat problematic for Windows users, since the Windows compilers are proprietary.  The PyStan documentation has instructions for Windows users on installing PyStan after installing Anaconda Python 3:  [PyStan on Windows — PyStan documentation](http://pystan.readthedocs.io/en/latest/windows.html).  Note that the instructions are for Windows 7 (the PyStan developers appear to dislike Windows 10).  In teaching a Cornell course, one of us (TL) has successfully followed this procedure (with an older compiler and Python/PyStan) with Windows 8.5.  If you have problems or discover useful tricks regarding PyStan on Windows before the workshop, please post your findings as an Issue on GitHub; perhaps other Windows users will be able to help, or benefit from your experience.  Our PyStan scripts are set up to automatically handle Windows limitations on parallel computing with Stan.

The two main software prerequisites are:

* A recent C/C++ compiler
* A recent Java JDK

Linux users will probably already have these (or know how to get them with their package manager if they don't).  

**Mac users** should do the following (we have tested this with macOS 10.11 and 10.12; the same setup will hopefull work with 10.13).

*Xcode and command-line tools for macOS:*

* Download Xcode using the Mac App Store app.  The current version is Xcode 9.2.  Our material runs on systems with Xcode 8 or Xcode 9, so you needn't upgrade to Xcode 9 if you already have Xcode 8 installed.  *Note that Xcode is large and the download can be time consuming—don't postpone this until the last minute.*
* *Launch Xcode.*  You must launch it after installing; it installs command-line tools after its first launch.
* Optional: Go back to the Mac App Store and check for updates. There may be a command-line tools update for Xcode.

*Java for macOS:*
Download and install a recent Java JDK (Java Development Kit). RStudio requires it.  On our machines, the environment works with both Java 8 and Java 9, so you shouldn't need to upgrade to Java 9 if you have Java 8 already installed.  You will find JDK installers here: [Java SE - Downloads | Oracle Technology Network](http://www.oracle.com/technetwork/java/javase/downloads/index.html).  **Note:** That site says that end users running but not developing with Java only need the JRE (Java Runtime Environment).  *This is not true.*  The exception is explained in FAQ #12 here: [Installation of the JDK and the JRE on macOS](https://docs.oracle.com/javase/9/install/installation-jdk-and-jre-macos.htm#JSJIG-GUID-2432241F-9517-4C0B-9CBB-489E6419C9C9). Anaconda's RStudio installation needs the JDK.  If you have Java 8 and use this occasion to upgrade to Java 9, you will find Java 8 uninstallation instructions here: [How do I uninstall Java on my Mac?](https://www.java.com/en/download/help/mac_uninstall_java.xml).

*XQuartz:* We recommend that Mac users who do not have the newest XQuartz install it or upgrade their current installation.  We do not know if it is a requirement for our environment, but `matplotlib` does interact with it (for fonts) if it is present. There are minor issues between the newest `matplotlib` and older versions of XQuartz; install the newest XQuartz from: [XQuartz](https://www.xquartz.org/).

**All users:**

* Install Anaconda with Python 3.6: [Downloads | Anaconda](https://www.anaconda.com/download/#macos).  Exceptions:

    - If you already have Anaconda installed, you do not need to re-install it, even if you are using an older Python version.
    - If you have a customized command-line environment and you'd like a setup with minimal impact in your default environment, install Miniconda with Python 3.6 instead: [Miniconda — Conda](https://conda.io/miniconda.html).

* Update the `conda` command-line package manager in a terminal/shell session by running `conda update conda`.  **Note:** There is an important bug in the current Anaconda/Miniconda versions of `conda`, so unless you've done this update very recently, you *must* do it now.  The update command may report that it is changing your Anaconda package to a "custom" version.  This will not be a problem; when Anaconda is updated, your installation will eventually synchronize with it.

* Define and install the `cospop18` environment by using `conda` at the command line as follows (here "$" represents the prompt):

```
$ conda create -n CosPop18 python=3.6 anaconda pystan r-essentials rstudio rpy2 r-reticulate
```

* You may omit `rpy2` and `r-reticulate` in that command if you wish. These provide access to R from Python ([RPy2](https://rpy2.readthedocs.io/en/version_2.8.x/)) and to Python from R ([Reticulate](https://rstudio.github.io/reticulate/)).  We don't plan to use them during the workshop, but since we will use Python and R, you may want to experiment with them.  During installation, you may see instructions displayed about setting up `dbus`; this is a Linux application communication tool that we won't be using, so you may safely ignore these instructions.

* Verify that the environment works by activating it.  Activation runs a shell script that changes environment variables in the current shell.  This will likely display a lot of environment variable values on your terminal; this is fine (as long as there are no errors).

    -  On macOS and Linux, in your Terminal Window, use the shell's `source` command (which runs a script in the current shell session):
  ```
  $ source activate cospop18
  ```
    - On Windows, in your Anaconda Prompt, run:
  ```
  $ activate cospop18
  ```

* Deactivate the environment:  If you encounter a bug with this, you may not have properly updated `conda` (the recent bugfix mentioned above affects deativation).

    - On macOS and Linux, in your Terminal Window, run:
  `source deactivate`
    - On Windows, in your Anaconda Prompt, run:
  `deactivate`

At this point you should be set up.  If you'd like to run a couple quick tests, try these:

* Test PyStan using the Python interpreter by running these commands (you can copy and paste them at the Python or IPython interpreter prompt):

```python
import pystan

model_code = 'parameters {real y;} model {y ~ normal(0,1);}'

# The following will invoke Stan to build and compile a C++
# library; it will take some time and report progress to
# the console.
model = pystan.StanModel(model_code=model_code)

# This will run an MCMC algorithm; it will report progress
# to the console.
y = model.sampling(n_jobs=1).extract()['y']

y.mean()  # with luck the result will be near 0
```

* Launch RStudio:  Simply enter `rstudio` at a command-line prompt. This should launch the RStudio IDE; you may quit it after launch.
