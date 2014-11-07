from __future__ import with_statement
from fabric.api import cd, env, local, settings, run, sudo
from fabric.colors import green, red
from fabric.contrib.console import confirm

def prod_deploy(user='ubuntu',speed='fast'):
  print(red('Deploying Frontend to production at robobrain.me...'))
  if not confirm('Are you sure you want to deploy to production?'):
     print(red('Aborting deploy.'))
  env.host_string = 'ec2-54-68-27-137.us-west-2.compute.amazonaws.com'
  env.key_filename = 'conf/www.pem'
  env.user = user
  env.shell = '/bin/zsh -l -c'
  with cd('/var/www/Frontend'):
    print(green('Checking out test...'))
    run('git checkout test')
    print(green('Pulling latest version of test...'))
    run('git pull --no-edit origin test')
    print(green('Checking out production...'))
    run('git checkout production')
    print(green('Rebasing onto test...'))
    run('git rebase test --force-rebase')
    print(green('Pulling production...'))
    run('git pull --no-edit origin production')
    print(green('Pushing production upstream...'))
    run('git push origin production')
    if speed == 'fast':
      print(green('Building grunt without optimizing images...'))
      sudo('grunt prod_build --fast')
    else:
      print(green('Building grunt... (this usually takes 8 minutes)'))
      sudo('grunt prod_build')
      print(green('Cleaning serve directory... '))
      run('rm -R serve/*')
    print(green('Copying changes into ./serve/...'))
    run('cp -R dist/* serve/')
  print(red('Done!'))

def test_deploy(user='ubuntu', speed='fast'):
  env.host_string = 'ec2-54-218-20-10.us-west-2.compute.amazonaws.com'
  env.key_filename = 'conf/www.pem'
  env.user = user
  env.shell = '/bin/zsh -l -c'
  print(red('Deploying Frontend to test at test.robobrain.me...'))
  with cd('/var/www/Frontend'):
    print(green('Checking out master...'))
    run('git checkout master')
    print(green('Pulling latest version of master...'))
    run('git pull --no-edit origin master')
    print(green('Checking out test...'))
    run('git checkout test')
    print(green('Rebasing onto master...'))
    run('git rebase master test')
    print(green('Pulling latest version of test...'))
    run('git pull --no-edit origin test')
    print(green('Push the latest version of test...'))
    run('git push origin test')
    if speed == 'fast':
      print(green('Building grunt without optimizing images...'))
      sudo('grunt test_build --fast')
    else:
      print(green('Building grunt... (this usually takes 8 minutes)'))
      sudo('grunt test_build')
      print(green('Cleaning serve directory... '))
      run('rm -R serve/*')
    print(green('Copying changes into ./serve/...'))
    run('cp -R dist/* serve/')
  print(red('Done!'))
