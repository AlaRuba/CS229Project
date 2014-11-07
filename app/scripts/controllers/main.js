'use strict';

/**
 * @ngdoc function
 * @name roboBrainApp.controller:MainCtrl
 * @description
 * # MainCtrl
 * Controller of the roboBrainApp
 * Note: Feed type must be one of: affordance, vision, planning, object-usage, 3D-detection
 * Media content type must be one of: image
 */
angular.module('roboBrainApp')
  .controller('MainCtrl', ['$scope', 'brainFeeds', function ($scope, brainFeeds, ENV) {
    brainFeeds.init();
    $scope.feeds = brainFeeds.init();
    $scope.moreRecent = brainFeeds.moreRecent;
    $scope.vote = brainFeeds.vote;
    $scope.loadComments = brainFeeds.loadComments;
  }]);
