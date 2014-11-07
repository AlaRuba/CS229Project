'use strict';

/**
 * @ngdoc function
 * @name roboBrainApp.controller:AboutCtrl
 * @description
 * # AboutCtrl
 * Controller of the roboBrainApp
 */
angular.module('roboBrainApp')
  .controller('AboutCtrl', function ($scope) {
    $scope.news = [
    {
      name: 'New York Times',
      logo: 'images/news/nyt.jpg',
      url: 'http://www.nytimes.com/2014/09/02/science/robot-touch.html'
    },
    {
      name: 'TechCrunch',
      logo: 'images/news/tc.jpg',
      url: 'http://techcrunch.com/2014/08/25/the-robo-brain-will-use-the-internet-to-teach-robots/'
    },
    {
      name: 'Yahoo',
      logo: 'images/news/yahoo.jpg',
      url: 'https://www.yahoo.com/tech/cornell-scientists-create-robo-brain-to-teach-robots-95740298304.html'
    },
    {
      name: 'ABC',
      logo: 'images/news/abc.jpg',
      url: 'http://abcnews.go.com/Technology/robots-learn-teach/story?id=25196591'
    },
    {
      name: 'Kurzweil',
      logo: 'images/news/kurzweil.jpg',
      url: 'http://www.kurzweilai.net/robo-brain-will-teach-robots-everything-from-the-internet'
    },
    {
      name: 'Wired',
      logo: 'images/news/wired.jpg',
      url: 'http://www.wired.com/2014/08/robobrain/'
    },
    {
      name: 'CNet',
      logo: 'images/news/cnet.jpg',
      url: 'http://www.cnet.com/news/robo-brain-is-learning-from-the-internet/'
    },
    {
      name: 'CBS',
      logo: 'images/news/cbs.jpg',
      url: 'http://www.cbsnews.com/news/robo-brain-to-teach-robots-about-the-human-world/'
    },
    {
      name: 'Washington Post',
      logo: 'images/news/wpost.jpg',
      url: 'http://www.washingtonpost.com/news/speaking-of-science/wp/2014/08/25/this-robot-is-using-youtube-videos-to-learn-all-about-us/'
    },
    {
      name: 'BBC',
      logo: 'images/news/bbc.jpg',
      url: 'http://www.bbc.com/news/technology-28936436'
    },
    {
      name: 'Sky',
      logo: 'images/news/sky.jpg',
      url: 'http://news.sky.com/story/1325005/robo-brain-lets-robots-learn-from-the-internet'
    },
    {
      name: 'Daily Mail',
      logo: 'images/news/dmail.jpg',
      url: 'http://www.dailymail.co.uk/sciencetech/article-2734123/The-robot-brain-rule-Plans-giant-central-brain-power-millions-machines-world-revealed.html'
    },
    {
      name: 'Popular Science',
      logo: 'images/news/psci.jpg',
      url: 'http://www.popsci.com/blog-network/zero-moment/robo-brain-project-wants-turn-internet-robotic-hivemind'
    },
    {
      name: 'VOA',
      logo: 'images/news/voa.jpg',
      url: 'http://blogs.voanews.com/science-world/2014/08/25/12571/'
    }
    ]
    $scope.sponsors_gov = [
    {
      name: 'National Science Foundation',
      logo: 'images/sponsors/nsf.jpg',
      url: 'http://www.nsf.gov/'
    },
    {
      name: 'Office of Naval Research',
      logo: 'images/sponsors/onr.jpg',
      url: 'http://www.onr.navy.mil/'
    },
    {
      name: 'Army Research Office',
      logo: 'images/sponsors/aro.jpg',
      url: 'http://www.nsf.gov/'
    },
    {
      name: 'National Robotics Initiative',
      logo: 'images/sponsors/nri.jpg',
      url: 'http://www.nsf.gov/funding/pgm_summ.jsp?pims_id=503641&org=CISE'
    }];
  $scope.sponsors_tech = [
    {
      name: 'Google',
      logo: 'images/sponsors/goog.jpg',
      url: 'http://www.google.com/'
    },
    {
      name: 'Microsoft',
      logo: 'images/sponsors/msft.jpg',
      url: 'http://www.microsoft.com/'
    },
    {
      name: 'Qualcomm',
      logo: 'images/sponsors/qcom.jpg',
      url: 'http://www.qualcomm.com/'
    }];
  $scope.sponsors_misc = [
    {
      name: 'Alfred P. Sloan Foundation',
      logo: 'images/sponsors/apsf.jpg',
      url: 'http://www.sloan.org/'
    }];
  $scope.sponsors_schools = [
    {
      name: 'Cornell University',
      logo: 'images/sponsors/cornell.jpg',
      url: 'http://www.cornell.edu/'
    },
    {
      name: 'Stanford University',
      logo: 'images/sponsors/stanford.jpg',
      url: 'http://www.stanford.edu/'
    },
    {
      name: 'UC Berkeley',
      logo: 'images/sponsors/berkeley.jpg',
      url: 'http://www.berkeley.edu/'
    },
    {
      name: 'Brown University',
      logo: 'images/sponsors/brown.jpg',
      url: 'http://www.brown.edu/'
    }];
  });