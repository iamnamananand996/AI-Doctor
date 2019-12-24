$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result_data').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    // $('#btn-predict').click(function () {
    //     var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        // $(this).hide();
        // $('.loader').show();

        // Make prediction by calling api /predict
        // $.ajax({
        //     type: 'POST',
        //     url: '/predict_malaria_watson',
        //     data: form_data,
        //     contentType: false,
        //     cache: false,
        //     processData: false,
        //     async: true,
        //     success: function (resp) {
        //         console.log(resp)
        //         Get and display the result
        //         $('.loader').hide();
        //         $('#result').fadeIn(600);
        //         // $('#result').text(' Result:  ' + resp);
        //         console.log('Success!');
        //         console.log(resp.split(":"));
        //         var res = resp.split(":");
        //         if (Number(res[0]) > Number(res[1])){
        //             console.log(res[0])
        //             $('#result').text('Person is Parasitized');
        //         } 
        //         else{
        //             $('#result').text('Person is Uninfected');
        //         }
        //         $('.loader').hide();
        //         $('#result').fadeIn(600);
        //         $('#result_data').show();
        //         console.log('Success!');
        //         $('#not_ill').text(res[1]);
        //         $('#ill').text(res[0]);
        //     },
        // });
    // });

});
