

# Introduction to Linux

## Preparation

1. Boot from a usb stick (or live cd), we suggest to use  [Ubuntu gnome](http://ubuntugnome.org/) distribution, or another ubuntu derivative.

2. (Optional) Configure keyboard layout and software repository
   Go to the the *Activities* menu (top left corner, or *start* key):
      -  Go to settings, then keyboard. Set the layout for latin america
      -  Go to software and updates, and select the server for Colombia
3. (Optional) Instead of booting from a live Cd. Create a partition in your pc's hard drive and install the linux distribution of your choice, the installed Os should perform better than the live cd.

## Introduction to Linux

1. Linux Distributions

   Linux is free software, it allows to do all sort of things with it. The main component in linux is the kernel, which is the part of the operating system that interfaces with the hardware. Applications run on top of it. 
   Distributions pack together the kernel with several applications in order to provide a complete operating system. There are hundreds of linux distributions available. In
   this lab we will be using Ubuntu as it is one of the largest, better supported, and user friendly distributions.


2. The graphical interface

   Most linux distributions include a graphical interface. There are several of these available for any taste.
   (http://www.howtogeek.com/163154/linux-users-have-a-choice-8-linux-desktop-environments/).
   Most activities can be accomplished from the interface, but the terminal is where the real power lies.

### Playing around with the file system and the terminal
The file system through the terminal
   Like any other component of the Os, the file system can be accessed from the command line. Here are some basic commands to navigate through the file system

   -  ``ls``: List contents of current directory
   - ``pwd``: Get the path  of current directory
   - ``cd``: Change Directory
   - ``cat``: Print contents of a file (also useful to concatenate files)
   - ``mv``: Move a file
   - ``cp``: Copy a file
   - ``rm``: Remove a file
   - ``touch``: Create a file, or update its timestamp
   - ``echo``: Print something to standard output
   - ``nano``: Handy command line file editor
   - ``find``: Find files and perform actions on it
   - ``which``: Find the location of a binary
   - ``wget``: Download a resource (identified by its url) from internet 

Some special directories are:
   - ``.`` (dot) : The current directory
   -  ``..`` (two dots) : The parent of the current directory
   -  ``/`` (slash): The root of the file system
   -  ``~`` (tilde) :  Home directory
      
Using these commands, take some time to explore the ubuntu filesystem, get to know the location of your user directory, and its default contents. 
   
To get more information about a command call it with the ``--help`` flag, or call ``man <command>`` for a more detailed description of it, for example ``man find`` or just search in google.


## Input/Output Redirections
Programs can work together in the linux environment, we just have to properly 'link' their outputs and their expected inputs. Here are some simple examples:

1. Find the ```passwd```file, and redirect its contents error log to the 'Black Hole'
   >  ``find / -name passwd  2> /dev/null``

   The `` 2>`` operator redirects the error output to ``/dev/null``. This is a special file that acts as a sink, anything sent to it will disappear. Other useful I/O redirection operations are
      -  `` > `` : Redirect standard output to a file
      -  `` | `` : Redirect standard output to standard input of another program
      -  `` 2> ``: Redirect error output to a file
      -  `` < `` : Send contents of a file to standard input
      -  `` 2>&1``: Send error output to the same place as standard output

2. To modify the content display of a file we can use the following command. It sends the content of the file to the ``tr`` command, which can be configured to format columns to tabs.

   ```bash
   cat milonga.txt | tr '\n' ' '
   ```
   
## SSH - Server Connection

1. The ssh command lets us connect to a remote machine identified by SERVER (either a name that can be resolved by the DNS, or an ip address), as the user USER (**vision** in our case). The second command allows us to copy files between systems (you will get the actual login information in class).

   ```bash
   
   #connect
   ssh USER@SERVER
   ```

2. The scp command allows us to copy files form a remote server identified by SERVER (either a name that can be resolved by the DNS, or an ip address), as the user USER. Following the SERVER information, we add ':' and write the full path of the file we want to copy, finally we add the local path where the file will be copied (remember '.' is the current directory). If we want to copy a directory we add the -r option. for example:

   ```bash
   #copy 
   scp USER@SERVER:~/data/sipi_images .
   
   scp -r USER@SERVER:/data/sipi_images .
   ```
   
   Notice how the first command will fail without the -r option

See [here](ssh.md) for different types of SSH connection with respect to your OS.

## File Ownership and permissions   

   Use ``ls -l`` to see a detailed list of files, this includes permissions and ownership
   Permissions are displayed as 9 letters, for example the following line means that the directory (we know it is a directory because of the first *d*) *images*
   belongs to user *vision* and group *vision*. Its owner can read (r), write (w) and access it (x), users in the group can only read and access the directory, while other users can't do anything. For files the x means execute. 
   ```bash
   drwxr-x--- 2 vision vision 4096 ene 25 18:45 images
   ```
   
   -  ``chmod`` change access permissions of a file (you must have write access)
   -  ``chown`` change the owner of a file
   
## Sample Exercise: Image database

1. Create a folder with your Uniandes username. (If you don't have Linux in your personal computer)

2. Copy *sipi_images* folder to your personal folder. (If you don't have Linux in your personal computer)

3.  Decompress the images (use ``tar``, check the man) inside *sipi_images* folder. 

4.  Use  ``imagemagick`` to find all *grayscale* images. We first need to install the *imagemagick* package by typing

    ```bash
    sudo apt-get install imagemagick
    ```
    
    Sudo is a special command that lets us perform the next command as the system administrator
    (super user). In general it is not recommended to work as a super user, it should only be used 
    when it is necessary. This provides additional protection for the system.
    
    ```bash
    find . -name "*.tiff" -exec identify {} \; | grep -i gray | wc -l
    ```
    
3.  Create a script to copy all *color* images to a different folder
    Lines that start with # are comments
       
      ```bash
      #!/bin/bash
      
      # go to Home directory
      cd ~ # or just cd

      # remove the folder created by a previous run from the script
      rm -rf color_images

      # create output directory
      mkdir color_images

      # find all files whose name end in .tif
      images=$(find sipi_images -name *.tiff)
      
      #iterate over them
      for im in ${images[*]}
      do
         # check if the output from identify contains the word "gray"
         identify $im | grep -q -i gray
         
         # $? gives the exit code of the last command, in this case grep, it will be zero if a match was found
         if [ $? -eq 0 ]
         then
            echo $im is gray
         else
            echo $im is color
            cp $im color_images
         fi
      done
      
      ```
      -  save it for example as ``find_color_images.sh``
      -  make executable ``chmod u+x`` (This means add Execute permission for the user)
      -  run ``./find_duplicates.sh`` (The dot is necessary to run a program in the current directory)
      

## Your turn

## SOLUTION


### 1. What is the ``grep``command?

- ''grep'' command (``global regular expression and print``), is used to look for inside of directories the lines that match a pattern. By default, the command prints the lines found in the standard output, that means, it can be seemed directly on the screen.

The basic syntax is: ``$grep "pattern" [Filename]``

### Example: Search all the words that start with "e" inside of a file named "Uniandes"

   ```bash
   grep "e*" Uniandes
   ```
There are other options to complement ``grep`` command:

-``c`` Instead of printing the lines that match, it shows the number of lines that match.

-``e`` Allows us to specify several search patterns.

-``r`` Searches recursively within all subdirectories of the current directory.

-``v`` Shows us the lines that do not match the desired pattern.

-``i`` Ignores the distinction between uppercase and lowercase.

-``n`` Number the lines in the output.

-``E`` Allows us to use regular expressions. Equivalent to use ``egrep``.

-``o`` Tells grep to show us only the part of the line that matches the pattern.

-``f`` Extracts the patterns from the file that we specify. File patterns should go one per line.

-``H`` Prints the name of the file with each match.

### Bibliography: 
	1. El comando grep. (n.d.). Retrieved February 5, 2019, from http://nereida.deioc.ull.es/~pcgull/ihiu01/cdrom/unix/unix1/contenido/node74.html

	2. Khan, A. (n.d.). Introducción a la señalización celular. Retrieved from https://es.khanacademy.org/science/biology/cell-signaling/mechanisms-of-cell-signaling/a/introduction-to-cell-signaling


### 2. What is the meaning of ``#!/bin/python`` at the start of scripts?

- The structure ``#!/bin/python`` at the beginning of scripts, it's a shebang for your command line about how it should interpreter a script. (1)

A shebang, also known as sha-bang, hashbang, pound-bang, or hash-pling, is the character sequence of (#!) at the beginning of a script, its function is to convert a file of Python code into an executable program. The operating system uses ``#!/bin/python`` (shebang) to find an appropriate program for running the code along all the time, in this case, to execute using Python. (2,3)

### Bibliography: 
	1. Wikipedia. (n.d.). Shebang (Unix).

	2. Lutz, M. (n.d.). Learning Python (4th ed.). O’REILLY.

	3. Thoma, M. (n.d.). What does #!/usr/bin/python mean? Retrieved February 5, 2019, from https://martin-thoma.com/what-does-usrbinpython-mean/
		

### 3. Download using ``wget`` the [*bsds500*](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500) image segmentation database, and decompress it using ``tar`` (keep it in you hard drive, we will come back over this data in a few weeks).

I work on Mac OS system, so at first instance ``wget`` must be installed. This command isn't by default. There are different ways to install ``wget``on Mac OS (1), However, the easiest is to implemnet a very useful tool called ``Homebrew``. This is a free and open source software  ``package management system`` that helps with the installation of different software on Apple's macOS operating system and Linux (2). This tool has been used widely in ``GitHub''. ``Homebrew`` had the largest number of new contributors on GitHub (3).  
 
Following, this is the structure to install ´´Homebrew´´in Mac OS, according its creator, Max Howell (2).

   ```bash
   /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

   ```
After being installed ``Homebrew``, we can use it as the tool to install other packages as ``wget``:

   ```bash
	$brew install wget
   ```
And that's all, ``wget`` is already installed! and now it can be used to download the image segmentation database (bsds500).

The form to get the database is (4):

   ```bash
	$wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
   ```

After that, the file is decompressed (5):

   ```bash
	$tar zxf BSR_bsds500.tgz
   ```

### Bibliography: 
	1. HACKSPARROW. (n.d.). How to install wget on your Mac. Retrieved February 6, 2019, from https://www.hacksparrow.com/how-to-install-wget-on-your-mac.html

	2. Howell, M. (n.d.). Homebrew. Retrieved February 6, 2019, from https://brew.sh/

	3. Wikipedia. (n.d.). Homebrew (package management software).

	4. Berkeley. (n.d.). UC Berkeley Computer Vision Group - Contour Detection and Image Segmentation - Resources. Retrieved February 6, 2019, from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500

	5. Delgado, F. (n.d.). Comandos Unix / Linux – Guía de Referencia.


### 4. What is the disk size of the uncompressed dataset, How many images are in the directory 'BSR/BSDS500/data/images'?

### a). Disk size of the uncompressed dataset

We can use the command ``du -sh *`` to get the size. `s` refers to show only the information about size and `h` y to show them friendly (1). The uncompressed dataset was **BSR**, so the code sintaxis is:

   ```bash
	$ du -sh BSR
   ```
The result was: **72 MB**


### b). Number of Images

The form to find the number of files is using the command ``wc`` (**Word Count**). Depending of the use, there are some structures:

``wc -l`` <file> number of lines

``wc -c`` <file> number of bytes

``wc -m`` <file> prints the number of characters

``wc -L`` <file> prints the length of the longest line

``wc -w`` <file> prints the number of words

In our case, we can have other files that are not images, so it's important to find only the images. The filter to search the images is the command ``find``. This command is a very powerful tool to find files and directories (2). 

The sintaxis of find is:

   ```bash
	find [route] [expression of searching] [action]
   ```
There is another powerful command: ``-exec`` that lets implement other actions on the initial find, so in my case, i use the command ``identify`` from **ImageMagick** that describes the format and characteristics of one or more image files (3).  


The code to find only the images from the route ``BSR/BSDS500/data/images``
   ```bash
	find . -name "*jpg" -exec identify {} \;| wc -l
   ```
The result was: **500 Images**

### Bibliography: 
	1. Cantero, G. (n.d.). du Command. Retrieved February 7, 2019, from https://www.galisteocantero.com/como-ver-el-tamano-de-archivos-desde-la-consola-en-linux/

	2. Linuxtoal. (n.d.). Encuentra cualquier cosa en Linux con find. Retrieved February 7, 2019, from https://www.linuxtotal.com.mx/index.php?cont=info_admon_022

	3. @ ImageMagick. (n.d.). Command-line Tools: Identify. Retrieved February 7, 2019, from https://imagemagick.org/script/identify.php



 
### 5. What are all the different resolutions? What is their format? Tip: use ``awk``, ``sort``, ``uniq`` 

The resolution can be understand as the total number of pixels in the image by rows and columns. According with the example showed in the section of **Format and Print Image Properties, formats** of **ImageMagick page (4)**, the code to find the resolution and format of the images is:

   ```bash
	find . -name "*jpg" -exec identify -format "%m:%f %wx%h" {} \;
   ```
The result was:

   ```bash
	481x321JPEG;
	321x481JPEG
   ```	 
 
### Bibliography: 
	1. ImageMagick. (n.d.). Format and Print Image Properties. Retrieved February 7, 2019, from https://imagemagick.org/script/escape.php




### 6. How many of them are in *landscape* orientation (opposed to *portrait*)? Tip: use ``awk`` and ``cut``

As we can saw in the last question, there are 2 kind of images. The first is a landscape orientation of size 481x321 and the second one is a portrait orientation of size 321x481. 

### a) Landscape

   ```bash
	find . -name "*jpg" -exec identify -format " %wx%h\n" {} \; | grep '481x321' | wc -l
   ```	
**Answer = 348**

### b) Portrait

   ```bash
	find . -name "*jpg" -exec identify -format " %wx%h\n" {} \; | grep '321x481' | wc -l
   ```	 
**Answer = 152**



 
### 7. Crop all images to make them square (256x256) and save them in a different folder. Tip: do not forget about  [imagemagick](http://www.imagemagick.org/script/index.php).

At first, we have to create a new directory called ("Recortadas") where the new images are going to be saved. The code was:

   ```bash
	cp -r  Documents/IBIO4490/BSR/BSDS500/data/images  Documents/IBIO4490/Recortadas/
   ```

At second, with the command ``mortify`` of **ImageMagick** each image is cropped. It's important to use ``!`` to get the size exactly at 255x255. The code was: 

   ```bash
	mogrify -resize 256x256! *.jpg
   ```



# Report

For every question write a detailed description of all the commands/scripts you used to complete them. DO NOT use a graphical interface to complete any of the tasks. Use screenshots to support your findings if you want to. 

Feel free to search for help on the internet, but ALWAYS report any external source you used.

Notice some of the questions actually require you to connect to the course server, the login instructions and credentials will be provided on the first session. 

## Deadline

We will be delivering every lab through the [github](https://github.com) tool (Silly link isn't it?). According to our schedule we will complete that tutorial on the second week, therefore the deadline for this lab will be specially long **February 7 11:59 pm, (it is the same as the second lab)** 

### More information on

http://www.ee.surrey.ac.uk/Teaching/Unix/ 




