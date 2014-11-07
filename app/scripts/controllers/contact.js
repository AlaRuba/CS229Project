'use strict';

/**
 * @ngdoc function
 * @name roboBrainApp.controller:ContactCtrl
 * @description
 * # ContactCtrl
 * Controller of the roboBrainApp
 */
angular.module('roboBrainApp')
  .controller('ContactCtrl', function ($scope) {
    $scope.contacts = [
    {
      name: 'Ashutosh Saxena',
      email: 'asaxena@cs.cornell.edu',
      address: 'Gates 138',
      street: 'Stanford University',
      state: 'CA',
      zip: '94305'
    },
    {
      name: 'Aditya Jami',
      email: 'adityajami@gmail.com'
    }];
  });
