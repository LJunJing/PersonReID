const titleList = document.querySelectorAll(".menu-title");
Array.from(titleList).forEach((titleEle) => {
  titleEle.addEventListener("click", function () {
    titleEle.parentElement.classList.toggle("active");
    console.log(123);
  });
});

function uploadImage() {
  var get_img = document.getElementById("input_img");
  get_img.onchange = function () {
    var file = this.files;
    // console.log(file);
    var reader = new FileReader();
    reader.readAsDataURL(file[0]);
    reader.onload = function () {
      var image = document.createElement("img");
      image.width = "1000";
      image.src = reader.result;
      var showPicture = document.getElementById("show_img");
      showPicture.append(image);
    };
  };
}

