angular
  .module('roboBrainApp')
  .config(['$httpProvider', function($httpProvider) {
    $httpProvider.defaults.withCredentials = true;
    $httpProvider.defaults.xsrfCookieName = 'csrftoken';
    $httpProvider.defaults.xsrfHeaderName = 'X-CSRFToken';
   }])
  .factory('brainFeeds', ['$resource', '$interval', '$cookieStore', '$cookies', '$http', '$q', 'ENV', function($resource, $interval, $cookieStore, $cookies, $http, $q, ENV) {
    // initialize with k most recent feeds
    var feeds = {};
    var feedSize = 10;
    var offset = 0;
    var isScrolling = true;
    var votelock = false;
    var ordered_feeds = [];
    return {
      init: function() {
        feeds = {};
        offset = 0;
        votelock = false;
        ordered_feeds = [];
        return ordered_feeds;
      },
      moreRecent: function() {
        var feed_endpoint = ENV.apiEndpoint + 'feed/infinite_scroll/';
        var upvotesEndpoint = ENV.apiEndpoint + 'feed/upvotes/';
        if (isScrolling) {
          isScrolling = false;
          var val = $cookies.csrftoken;
          if (!val) {
            $http.get(upvotesEndpoint);
          }
          $http.get(feed_endpoint, {
            params: {
              k: feedSize,
              cur: offset
            }
          })
          .then(function(response) {
            var data = response.data;
            for (var k = 0; k < data.length; ++k) {
              var feed_id = data[k]['_id'];
              feeds[feed_id] = data[k];
              feeds[feed_id]['order'] = offset;
              ordered_feeds[offset] = feeds[feed_id];
              offset += 1;
              if ($cookieStore.get(feed_id) == 1) {
                feeds[feed_id]['upvoted'] = true;
              } else if ($cookieStore.get(feed_id) == -1) {
                feeds[feed_id]['downvoted'] = true;
              }
              feeds[feed_id]['text'] = feeds[feed_id]['text'].split(' ');
              for (var j = 0; j < feeds[feed_id]['media'].length; ++j) {
                // Bundle tellmedave robot sequence with simulator url
                if (feeds[feed_id]['media'][j].match(/\.(gif|jpg|jpeg|tiff|png)$/i)) {
                  var p = feeds[feed_id]['media'][j].lastIndexOf('.');
                  feeds[feed_id]['media'][j] = {
                    type: 'image',
                    url: ENV.staticEndpoint + feeds[feed_id]['media'][j].slice(0, p) + '_thumb' + feeds[feed_id]['media'][j].slice(p)
                  };
                }
                else if (feeds[feed_id]['media'][j].match(/\.(htm|html)$/i)) {
                  feeds[feed_id]['media'][j] = {
                    type: 'html',
                    url: ENV.staticEndpoint + feeds[feed_id]['media'][j],
                  };
                }
                else if (feeds[feed_id]['media'][j].match(/\.(mp4)$/i)) {
                  feeds[feed_id]['media'][j] = {
                    type: 'mp4',
                    url: ENV.staticEndpoint + feeds[feed_id]['media'][j],
                  };
                }
                else if (feeds[feed_id]['media'][j].match(/\.(webm)$/i)) {
                  feeds[feed_id]['media'][j] = {
                    type: 'webm',
                    url: ENV.staticEndpoint + feeds[feed_id]['media'][j],
                  };
                }
              }
            }
            isScrolling = true;
          });
        }
      },
      vote: function(feedid, direction) {
        var upvotesEndpoint = ENV.apiEndpoint + 'feed/upvotes/';
        var change = false;

        if (votelock) {
          return;
        }
        // If a vote has already be made, only register it if it's a change vote
        if ($cookieStore.get(feedid)) {
          if ($cookieStore.get(feedid) == direction) {
            return;
          } else {
            change = true;
          }
        }
        votelock = true;

        $q.when((function() {
          var val = $cookies.csrftoken;
          if (!val) {
            return $http.get(upvotesEndpoint);
          }
          return "Ok";
        })())
        .then(function(response) {
          $http.defaults.headers.post['X-CSRFToken'] = $cookies['csrftoken'];
          return $http.post(upvotesEndpoint, {
            feedid: feedid,
            vote: direction,
            change: change
          })
        })
        .then(function(response) {
          response = response.data;
          $cookieStore.put(feedid, direction);
          if (direction == 1) {
            feeds[feedid]['upvoted'] = true
            feeds[feedid]['downvoted'] = false
          } else if (direction == -1) {
            feeds[feedid]['upvoted'] = false
            feeds[feedid]['downvoted'] = true
          }
          feeds[feedid]['upvotes'] = response['upvotes']
          feeds[feedid]['downvotes'] = response['downvotes']
          votelock = false;
        });
      },
      loadComments: function(feed_id, sourceText) {
        if ($('#iframe-comment-'+feed_id).length == 0) {
          var iframe = document.createElement('iframe');
          iframe.src = 'static/comments.html?id=' + encodeURIComponent(sourceText);
          iframe.className = 'feed-iframe-comments'
          iframe.frameBorder = 0;
          iframe.id = 'iframe-comment-'+feed_id;
          $('#'+feed_id).append(iframe);
        } else {
          var iframeElement = $('#iframe-comment-'+feed_id);
          var currStyle = iframeElement.css('display');
          if (currStyle == 'block') {
            iframeElement.css('display', 'none');
          } else if (currStyle == 'none') {
            iframeElement.css('display', 'block');
          }
        }
      },
      query: function(query) {
        // var feeds = {};
        var query_endpoint = '';
        if (query['hashword']) {
          query_endpoint = ENV.apiEndpoint + 'feed/filter/'
        } else if (query['type']) {
          query_endpoint = ENV.apiEndpoint + 'feed/filter_type/'
        } else {
          return;
        }
        query['k'] = feedSize;
        var val = $cookies.csrftoken;
        if (!val) {
          $http.get(upvotesEndpoint);
        }
        $http.get(query_endpoint, {
          params: query
        }).then(function(response) {
          var data = response.data;
          for (var i = 0; i < data.length; ++i) {
            var feed_id = data[i]['_id'];
            feeds[feed_id] = data[i];
            feeds[feed_id]['order'] = i;
            if ($cookieStore.get(feed_id) == 1) {
              feeds[feed_id]['upvoted'] = true;
            } else if ($cookieStore.get(feed_id) == -1) {
              feeds[feed_id]['downvoted'] = true;
            }
            feeds[feed_id]['text'] = feeds[feed_id]['text'].split(' ');
            for (var j = 0; j < feeds[feed_id]['media'].length; ++j) {
              // Bundle tellmedave robot sequence with simulator url
              if (feeds[feed_id]['media'][j].match(/\.(gif|jpg|jpeg|tiff|png)$/i)) {
                var p = feeds[feed_id]['media'][j].lastIndexOf('.');
                feeds[feed_id]['media'][j] = {
                  type: 'image',
                  url: ENV.staticEndpoint + feeds[feed_id]['media'][j].slice(0, p) + '_thumb' + feeds[feed_id]['media'][j].slice(p)
                };
              }
              else if (feeds[feed_id]['media'][j].match(/\.(htm|html)$/i)) {
                feeds[feed_id]['media'][j] = {
                  type: 'html',
                  url: ENV.staticEndpoint + feeds[feed_id]['media'][j]
                };
              }
              else if (feeds[feed_id]['media'][j].match(/\.(mp4)$/i)) {
                feeds[feed_id]['media'][j] = {
                  type: 'mp4',
                  url: ENV.staticEndpoint + feeds[feed_id]['media'][j],
                };
              }
              else if (feeds[feed_id]['media'][j].match(/\.(webm)$/i)) {
                feeds[feed_id]['media'][j] = {
                  type: 'webm',
                  url: ENV.staticEndpoint + feeds[feed_id]['media'][j],
                };
              }
            }
          }
        });
      }
    };
  }]);
