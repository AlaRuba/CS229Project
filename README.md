RoboBrain Frontend v.0.02
====================

RoboBrain website for the web. Includes an
introduction to RoboBrain, its contributors,
and the publications associated with the
project.


### Dependencies

* compass (ruby gem)
* npm Package Manager (for testing only)
* Bower Package Manager


### Installation

In your terminal, navigate to this project's
root folder. Run "npm install" and "bower install"
to install all remaining dependencies.


### Testing

To test the frontend on localhost, navigate
to this project's root folder in your terminal
and run "grunt serve".


### Deployment

The RoboBrain website can be served using any
static file server (i.e. Apache, nginx). Run "grunt build",
then copy the contents of the new "dist" folder to
the public directory where Apache or nginx is
listening.