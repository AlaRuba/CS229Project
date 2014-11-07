'use strict';

/**
 * @ngdoc function
 * @name roboBrainApp.controller:ResearchCtrl
 * @description
 * # ResearchCtrl
 * Controller of the roboBrainApp
 */
angular.module('roboBrainApp')
  .controller('ResearchCtrl', function ($scope) {
    $scope.publications = [
    {
      title: 'Deep Learning for Detecting Robotic Grasps',
      authors: 'Ian Lenz, Honglak Lee, Ashutosh Saxena',
      collection: 'International Journal on Robotics Research (IJRR)',
      year: '2014',
      url: 'http://pr.cs.cornell.edu/deepgrasping/',
      misc: 'An earlier version was presented in Robotics Science and Systems (RSS), 2013.'
    },
    {
      title: 'Tell Me Dave: Context-Sensitive Grounding of Natural Language to Mobile Manipulation Instructions',
      authors: 'Dipendra K. Misra, Jaeyong Sung, Kevin Lee, Ashutosh Saxena',
      collection: 'Robotics: Science and Systems (RSS)',
      year: '2014',
      url: 'http://tellmedave.cs.cornell.edu/paper.pdf'
    },
    {
      title: 'Hierarchical Semantic Labeling for Task-Relevant RGB-D Perception',
      authors: 'Chenxia Wu, Ian Lenz, Ashutosh Saxena',
      collection: 'Robotics: Science and Systems (RSS)',
      year: '2014',
      url: 'http://www.cs.cornell.edu/~asaxena/papers/wulenzsaxena_hierarchicallabel-rgbd_rss2014.pdf'
    },
    {
      title: 'Modeling High-Dimensional Humans for Activity Anticipation using Gaussian Process Latent CRFs',
      authors: 'Yun Jiang, Ashutosh Saxena',
      collection: 'Robotics: Science and Systems (RSS)',
      year: '2014',
      url: 'http://www.cs.cornell.edu/~asaxena/papers/jiang_saxena_rss14_gplcrf.pdf'
    },
    {
      title: 'Physically-Grounded Spatio-Temporal Object Affordances',
      authors: 'Hema S. Koppula, Ashutosh Saxena',
      collection: 'European Conference on Computer Vision (ECCV)',
      year: '2014',
      url: 'http://www.cs.cornell.edu/~asaxena/papers/koppula-saxena-eccv14-physicallygroundedaffordances.pdf'
    },
    {
      title: 'Anticipatory Planning for Human-Robot Teams',
      authors: 'Hema S. Koppula, Ashesh Jain, Ashutosh Saxena',
      collection: '14th International Symposium on Experimental Robotics (ISER)',
      year: '2014',
      url: 'http://www.cs.cornell.edu/~asaxena/papers/koppula_jain_saxena_iser14_anticipatoryplanning.pdf'
    },
    {
      title: '3D Reasoning from Blocks to Stability',
      authors: 'Zhaoyin Jia, Andy Gallagher, Ashutosh Saxena, Tsuhan Chen',
      collection: 'IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)',
      year: '2014',
      url: 'http://www.cs.cornell.edu/~asaxena/papers/3d_based_reasoning_saxena_pami14.pdf',
      misc: 'Original version appeared as an oral in CVPR 2013.'
    },
    {
      title: 'Learning Trajectory Preferences for Manipulators via Iterative Improvement',
      authors: 'Ashesh Jain, Brian Wojcik, Thorsten Joachims, Ashutosh Saxena',
      collection: 'Neural Information Processing Systems (NIPS)',
      year: '2013',
      url: 'http://www.cs.cornell.edu/~asaxena/papers/jain_saxena_trajectorypreferencelearning_nips13.pdf',
      misc: 'An earlier version of this work was presented at the ICML workshop on Robot Learning, June 2013.'
    },
    {
      title: 'Anticipating Human Activities using Object Affordances for Reactive Robotic Response',
      authors: 'Hema S. Koppula, Ashutosh Saxena',
      collection: 'Robotics: Science and Systems (RSS)',
      year: '2013',
      url: 'http://www.cs.cornell.edu/~asaxena/papers/anticipation_RSS_2013.pdf',
      awards: 'Best Student Paper Award, Best Paper Runner-Up'
    },
    {
      title: 'Infinite Latent Conditional Random Fields for Modeling Environments through Humans',
      authors: 'Yun Jiang, Ashutosh Saxena',
      collection: 'Robotics: Science and Systems (RSS)',
      year: '2013',
      url: 'http://www.cs.cornell.edu/~asaxena/papers/jiang_saxena_rss13_ILCRF.pdf'
    },
    {
      title: 'Learning Object Arrangements in 3D Scenes using Human Context',
      authors: 'Yun Jiang, Marcus Lim, Ashutosh Saxena',
      collection: 'International Conference of Machine Learning (ICML)',
      year: '2012',
      url: 'http://www.cs.cornell.edu/~asaxena/papers/jiang_saxena_rss13_ILCRF.pdf'
    },
    {
      title: 'Contextual Guided Semantic Labeling and Search for 3D Point Clouds',
      authors: 'Abhishek Anand, Hema S. Koppula, Thorsten Joachims, Ashutosh Saxena',
      collection: 'International Journal of Robotics Research (IJRR)',
      year: '2012',
      url: 'http://www.cs.cornell.edu/~asaxena/papers/contextuallabeling3d_saxena_ijrr12.pdf',
      misc: 'Original version in NIPS 2011.'
    },
    {
      title: 'Learning to Open New Doors',
      authors: 'Ellen Klingbeil, Ashutosh Saxena, Andrew Y. Ng',
      collection: 'International Conference on Intelligent Robots and Systems (IROS)',
      year: '2010',
      url: 'http://www.cs.cornell.edu/~asaxena/openingnewdoors/klingbeil-saxena-ng-opennewdoors.pdf',
      misc: 'First published in Robotics Science and Systems (RSS) workshop on Robot Manipulation, 2008.'
    },
    {
      title: 'Robotic Grasping of Novel Objects using Vision',
      authors: 'Ashutosh Saxena, Justin Driemeyer, Andrew Y. Ng',
      collection: 'International Journal of Robotics Research (IJRR)',
      year: 'vol. 27, no. 2, pp. 157-173, Feb. 2008',
      url: 'http://pr.cs.cornell.edu/grasping/IJRR_saxena_etal_roboticgraspingofnovelobjects.pdf',
      misc: 'Original version appeared in Neural Information Processing Systems (NIPS), 2006.'
    }];
  });
