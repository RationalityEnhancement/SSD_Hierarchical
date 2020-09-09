jsPsych.plugins['multi-response'] = (function(){

var plugin = {};

  plugin.trial = function(display_element, trial) {

    // default trial parameters
    display_element.html('');
    trial.button_html = trial.button_html || '<button class="jspsych-btn">%choice%</button>';
    trial.response_ends_trial = (typeof trial.response_ends_trial === 'undefined') ? true : trial.response_ends_trial;
    trial.timing_stim = trial.timing_stim || -1; // if -1, then show indefinitely
    trial.timing_response = trial.timing_response || -1; // if -1, then wait for response forever
    trial.is_html = (typeof trial.is_html === 'undefined') ? false : trial.is_html;
    trial.prompt = (typeof trial.prompt === 'undefined') ? "" : trial.prompt;

    // Multi choice
    var plugin_id_name = "jspsych-survey-multi-choice";
    var plugin_id_selector = '#' + plugin_id_name;
    var _join = function( /*args*/ ) {
      var arr = Array.prototype.slice.call(arguments, _join.length);
      return arr.join(separator = '-');
    }

    // Multi choice: trial defaults
    trial.preamble = typeof trial.preamble == 'undefined' ? "" : trial.preamble;
    trial.required = typeof trial.required == 'undefined' ? null : trial.required;
    trial.horizontal = typeof trial.required == 'undefined' ? false : trial.horizontal;

    // Text
    trial.preamble = typeof trial.preamble == 'undefined' ? "" : trial.preamble;
    if (typeof trial.rows == 'undefined') {
      trial.rows = [];
      for (var i = 0; i < trial.questions_text.length; i++) {
        trial.rows.push(5);
      }
    }
    if (typeof trial.columns == 'undefined') {
      trial.columns = [];
      for (var i = 0; i < trial.questions_text.length; i++) {
        trial.columns.push(80);
      }
    }

    // if any trial variables are functions
    // this evaluates the function and replaces
    // it with the output of the function
    // trial = jsPsych.pluginAPI.evaluateFunctionParameters(trial);  // Button response
    trial = jsPsych.pluginAPI.evaluateFunctionParameters(trial, protect=['on_mistake']);  // Multi choice

    // this array holds handlers from setTimeout calls
    // that need to be cleared if the trial ends early
    var setTimeoutHandlers = [];

    // display stimulus
    if (!trial.is_html) {
      display_element.append($('<img>', {
        src: trial.stimulus,
        id: 'jspsych-button-response-stimulus',
        class: 'block-center'
      }));
    } else {
      display_element.append($('<div>', {
        html: trial.stimulus,
        id: 'jspsych-button-response-stimulus',
        class: 'block-center'
      }));
    }

    // Multiple choice: form element
    var trial_form_id = _join(plugin_id_name, "form");
    display_element.append($('<form>', {
      "id": trial_form_id
    }));
    var $trial_form = $("#" + trial_form_id);

    // Multiple choice: show preamble text
    var preamble_id_name = _join(plugin_id_name, 'preamble');
    $trial_form.append($('<div>', {
      "id": preamble_id_name,
      "class": preamble_id_name
    }));
    $('#' + preamble_id_name).html(trial.preamble);

    for (var i = 0; i < trial.questions.length; i++) {
      // create question container
      var question_classes = [_join(plugin_id_name, 'question')];
      if (trial.horizontal) {
        question_classes.push(_join(plugin_id_name, 'horizontal'));
      }

      $trial_form.append($('<div>', {
        "id": _join(plugin_id_name, i),
        "class": question_classes.join(' ')
      }));

      var question_selector = _join(plugin_id_selector, i);

      // add question text
      $(question_selector).append(
        '<p class="' + plugin_id_name + '-text survey-multi-choice">' + trial.questions[i] + '</p>'
      );

      // create option radio buttons
      for (var j = 0; j < trial.options[i].length; j++) {
        var option_id_name = _join(plugin_id_name, "option", i, j),
          option_id_selector = '#' + option_id_name;

        // add radio button container
        $(question_selector).append($('<div>', {
          "id": option_id_name,
          "class": _join(plugin_id_name, 'option')
        }));

        // add label and question text
        var option_label = '<label class="' + plugin_id_name + '-text">' + trial.options[i][j] + '</label>';
        $(option_id_selector).append(option_label);

        // create radio button
        var input_id_name = _join(plugin_id_name, 'response', i);
        $(option_id_selector + " label").prepend('<input type="radio" name="' + input_id_name + '" value="' + trial.options[i][j] + '">');
      }

      if (trial.required && trial.required[i]) {
        // add "question required" asterisk
        $(question_selector + " p").append("<span class='required'>*</span>")

        // add required property
        $(question_selector + " input:radio").prop("required", true);
      }
    }

    // Text: show preamble text
    display_element.append($('<div>', {
      "id": 'jspsych-survey-text-preamble',
      "class": 'jspsych-survey-text-preamble'
    }));

    $('#jspsych-survey-text-preamble').html(trial.preamble);

    // Text: add questions
    for (var i = 0; i < trial.questions_text.length; i++) {
      // create div
      display_element.append($('<div>', {
        "id": 'jspsych-survey-text-' + i,
        "class": 'jspsych-survey-text-question'
      }));

      // add question text
      $("#jspsych-survey-text-" + i).append('<p class="jspsych-survey-text">' + trial.questions_text[i] + '</p>');

      // add text box
      $("#jspsych-survey-text-" + i).append('<textarea name="#jspsych-survey-text-response-' + i + '" cols="' + trial.columns[i] + '" rows="' + trial.rows[i] + '"></textarea>');
    }

    //display buttons
    var buttons = [];
    if (Array.isArray(trial.button_html)) {
      if (trial.button_html.length == trial.choices.length) {
        buttons = trial.button_html;
      } else {
        console.error('Error in button-response plugin. The length of the button_html array does not equal the length of the choices array');
      }
    } else {
      for (var i = 0; i < trial.choices.length; i++) {
        buttons.push(trial.button_html);
      }
    }
    display_element.append('<div id="jspsych-button-response-btngroup" class="center-content block-center"></div>')
    for (var i = 0; i < trial.choices.length; i++) {
      var str = buttons[i].replace(/%choice%/g, trial.choices[i]);
      $('#jspsych-button-response-btngroup').append(
        $(str).attr('id', 'jspsych-button-response-button-' + i).data('choice', i).addClass('jspsych-button-response-button').on('click', function(e) {
          var choice = $('#' + this.id).data('choice');
          after_response(choice);
        })
      );
    }

    //show prompt if there is one
    if (trial.prompt !== "") {
      display_element.append(trial.prompt);
    }

    // store response
    var response = {
      rt: -1,
      button: -1
    };

    // start time
    var start_time = 0;

    // function to handle responses by the subject
    function after_response(choice) {

      // measure rt
      var end_time = Date.now();
      var rt = end_time - start_time;
      response.button = choice;
      response.rt = rt;

      // after a valid response, the stimulus will have the CSS class 'responded'
      // which can be used to provide visual feedback that a response was recorded
      $("#jspsych-button-response-stimulus").addClass('responded');

      // disable all the buttons after a response
      $('.jspsych-button-response-button').off('click').attr('disabled', 'disabled');

      if (trial.response_ends_trial) {
        end_trial();
      }
    };

    // function to end trial when it is time
    function end_trial() {

      // kill any remaining setTimeout handlers
      for (var i = 0; i < setTimeoutHandlers.length; i++) {
        clearTimeout(setTimeoutHandlers[i]);
      }

      // Multi choice: create object to hold responses
      var question_data = [];
      $("div." + plugin_id_name + "-question").each(function(index) {
        var val = $(this).find("input:radio:checked").val();
        question_data.push(val)
      });

      var correct = null;
      var mistake = false;
      console.log(trial.correct);
      if (trial.correct) {
        correct = [];
        for (var i = 0; i < question_data.length; i++) {
          var c = trial.correct[i] == question_data[i];
          if (!c) mistake = true;
          correct.push(c)
        }
      }

      // Text: create object to hold responses
      var text_question_data = {};
      $("div.jspsych-survey-text-question").each(function(index) {
        var id = "Q" + index;
        var val = $(this).children('textarea').val();
        var obje = {};
        obje[id] = val;
        $.extend(text_question_data, obje);
      });

      // gather the data to store for the trial
      var trial_data = {
        // Button
        "rt": response.rt,
        "stimulus": trial.stimulus,
        "button_pressed": response.button,

        // Multiple choice
        // "rt": response_time,
        "questions": JSON.stringify(trial.questions),
        "responses": JSON.stringify(question_data),
        "correct": correct,

        // Text
        "text_responses": JSON.stringify(text_question_data)
      };

      // clear the display
      display_element.html('');

      // move on to the next trial
      jsPsych.finishTrial(trial_data);
    };

    // start timing
    start_time = Date.now();

    // hide image if timing is set
    if (trial.timing_stim > 0) {
      var t1 = setTimeout(function() {
        $('#jspsych-button-response-stimulus').css('visibility', 'hidden');
      }, trial.timing_stim);
      setTimeoutHandlers.push(t1);
    }

    // end trial if time limit is set
    if (trial.timing_response > 0) {
      var t2 = setTimeout(function() {
        end_trial();
      }, trial.timing_response);
      setTimeoutHandlers.push(t2);
    }

  };

  return plugin;
})();
