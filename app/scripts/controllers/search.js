'use strict';

/**
 * @ngdoc function
 * @name roboBrainApp.controller:SearchCtrl
 * @description
 * # SearchCtrl
 * Controller to query the knowledge graph
 */
angular.module('roboBrainApp')
  .controller('SearchCtrl', ['$scope', '$http', function ($scope, $http, ENV) {

    var search_endpoint = "http://test.robobrain.me:3000/graph_query/nlquery/"
    $scope.hasSearched = false;
    $scope.hasAnswered = false;
    $scope.hasError = false;

    $scope.searchGraph = function(){
        $scope.hasAnswered = false;
        $scope.hasError = false;
        $scope.hasSearched = true;

        var input = $scope.inputText;
        $scope.inputText = "";
        $scope.originalText = input;

        $http.get(search_endpoint, {
            params: {
                user_in: input,
            }
        }).success(function(data, status) {
            //console.log("SUCCESS");
            //console.log(data);
            $scope.outputText = data[0].user_out;
            $scope.hasSearched = false;
            $scope.hasAnswered = true;
        })
        .error(function(data, status) {
            //console.log("ERROR");
            $scope.hasSearched = false;
            $scope.hasError = true;
        });
    }

  }]);
