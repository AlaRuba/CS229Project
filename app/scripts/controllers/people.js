'use strict';

/**
 * @ngdoc function
 * @name roboBrainApp.controller:PeopleCtrl
 * @description
 * # PeopleCtrl
 * Controller of the roboBrainApp
 */
angular.module('roboBrainApp')
  .controller('PeopleCtrl', function ($scope) {

    $scope.leads = [
    {
      name: 'Prof. Ashutosh Saxena',
      institution: 'Cornell / Stanford',
      role: 'Robot Learning and Project Lead',
      photo: 'images/contrib/asaxena.jpg',
      url: 'http://www.cs.cornell.edu/~asaxena/'
    },
    {
      name: 'Ashesh Jain',
      institution: 'Cornell / Stanford',
      role: 'Interactive Online Learning',
      photo: 'images/contrib/ajain.jpg',
      url: 'http://www.cs.cornell.edu/~ashesh/',
      roleUrl: 'http://planit.cs.cornell.edu/'
    },
    {
      name: 'Ozan Sener',
      institution: 'Cornell / Stanford',
      role: 'Vision and Language',
      photo: 'images/contrib/osener.jpg',
      url: 'http://ozansener.net/'
    },
    {
      name: 'Dipendra K. Misra',
      institution: 'Cornell / Stanford',
      role: 'Robot Language (Tell Me Dave)',
      photo: 'images/contrib/dmisra.jpg',
      url: 'http://www.cs.cornell.edu/~dkm/',
      roleUrl: 'http://tellmedave.cs.cornell.edu/'
    },
    {
      name: 'Aditya Jami',
      institution: 'Cornell / Zoodig',
      role: 'Large-Scale Systems',
      photo: 'images/contrib/ajami.jpg',
      url: 'http://www.linkedin.com/in/adityajami'
    }];
    $scope.ugrad_masters_students = [{
      name: 'Debarghya Das',
      institution: 'Cornell University',
      role: 'Planning and Visualization',
      photo: 'images/contrib/ddas.jpg',
      url: 'http://debarghyadas.com/'
    },
    {
      name: 'Kevin Lee',
      institution: 'Cornell University',
      role: 'Online Robot Games',
      photo: 'images/contrib/klee.jpg',
      url: 'mailto:kkl53@cornell.edu'
    },
    {
      name: 'Michela Meister',
      institution: 'Stanford University',
      role: 'Language Queries',
      photo: 'images/contrib/mmeister.jpg',
      url: 'mailto:mmeister@stanford.edu'
    },
    {
      name: 'Hope Casey-Allen',
      institution: 'Stanford University',
      role: 'Knowledge Integration',
      photo: 'images/contrib/hcallen.jpg',
      url: 'mailto:hcaseyal@stanford.edu'
    },
    {
      name: 'Arzav Jain',
      institution: 'Stanford University',
      role: 'Graph Operations',
      photo: 'images/contrib/arjain.jpg',
      url: 'mailto:arzavj@stanford.edu'
    }
    ,{
      name: 'Gabriel Kho',
      institution: 'Stanford University',
      role: 'Human-Computer Interaction',
      photo: 'images/contrib/gkho.jpg',
      url: 'mailto:gabriel.d.kho@gmail.com'
    }
    ];


    $scope.faculty = [
    {
      name: 'Prof. Ken Goldberg',
      institution: 'UC Berkeley',
      role: 'Cloud Robotics',
      photo: 'images/contrib/kgoldberg.jpg',
      url: 'http://goldberg.berkeley.edu/'
      // roleUrl: 'http://goldberg.berkeley.edu/cloud-robotics/'
    },
    {
      name: 'Prof. Stefanie Tellex',
      institution: 'Brown University',
      role: 'Robot Dialog',
      photo: 'images/contrib/stellex.jpg',
      url: 'https://cs.brown.edu/people/stefie10/'
    },
    {
      name: 'Prof. Thorsten Joachims',
      institution: 'Cornell University',
      role: 'Machine Learning',
      photo: 'images/contrib/tjoachims.jpg',
      url: 'http://www.cs.cornell.edu/People/tj/'
    },
    {
      name: 'Prof. Silvio Savarese',
      institution: 'Stanford University',
      role: '3D Vision',
      photo: 'images/contrib/ssavarese.jpg',
      url: 'http://cvgl.stanford.edu/silvio/'
    },
    {
      name: 'Prof. Fei-Fei Li',
      institution: 'Stanford University',
      role: 'Computer Vision',
      photo: 'images/contrib/fli.jpg',
      url: 'http://vision.stanford.edu/feifeili/'
    },
    {
      name: 'Prof. Bart Selman',
      institution: 'Cornell University',
      role: 'Reasoning and AI',
      photo: 'images/contrib/bselman.jpg',
      url: 'http://www.cs.cornell.edu/selman/'
    },
    {
      name: 'Prof. Jitendra Malik',
      institution: 'UC Berkeley',
      role: 'Computer Vision',
      photo: 'images/contrib/jmalik.jpg',
      url: 'http://www.cs.berkeley.edu/~malik/'
    },
    {
      name: 'Prof. Ross Knepper',
      institution: 'Cornell University',
      role: 'Structured Planning',
      photo: 'images/contrib/rknepper.jpg',
      url: 'http://people.csail.mit.edu/rak/www/'
    }];


    $scope.phd_students = [
    {
      name: 'Hema Koppula',
      institution: 'Cornell University',
      role: 'Machine Learning and 3D Perception',
      photo: 'images/contrib/hkoppula.jpg',
      url: 'http://www.cs.cornell.edu/~hema/'
    },
    {
      name: 'Yun Jiang',
      institution: 'Cornell University',
      role: 'Non-Parametric Learning',
      photo: 'images/contrib/yjiang.jpg',
      url: 'http://www.cs.cornell.edu/~yunjiang/'
    },
    {
      name: 'Ian Lenz',
      institution: 'Cornell University',
      role: 'Deep Learning and Manipulation',
      photo: 'images/contrib/ilenz.jpg',
      url: 'http://www.cs.cornell.edu/~ianlenz/'
    },
    {
      name: 'Jaeyong Sung',
      institution: 'Cornell University',
      role: 'Manipulation Planning',
      photo: 'images/contrib/jsung.jpg',
      url: 'http://www.cs.cornell.edu/~jysung/'
    },
    {
      name: 'Chenxia Wu',
      institution: 'Cornell University',
      role: '3D Vision',
      photo: 'images/contrib/cwu.jpg',
      url: 'http://www.cs.cornell.edu/~chenxiawu/'
    },
    {
      name: 'Ayush Dubey',
      institution: 'Cornell University',
      role: 'Graph Databases (Weaver)',
      photo: 'images/contrib/adubey.jpg',
      url: 'http://www.cs.cornell.edu/~dubey'
    },
    {
      name: 'David Whitney',
      institution: 'Brown University',
      role: 'Semantics and Cooking',
      photo: 'images/contrib/dwhitney.jpg',
      url: 'https://cs.brown.edu/people/dwhitney/'
    }];
  });
