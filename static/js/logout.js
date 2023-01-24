const v_default = document.getElementById("v-default");
const v_signout = document.getElementById("v-signout");

v_signout.style.display = "none";

function signOut() {
	if (confirm("End your session?") == true) {
		v_default.style.display = "none";
		v_signout.style.display = "block";
		firebase
			.auth()
			.signOut()
			.then(() => {
				window.location = `/logout`;
			})
			.catch((error) => {
				v_signout.style.display = "none";
				v_default.style.display = "block";
				console.log(`error sign-out from index: ${error}`);
			});
	} else {
		console.log("ini cancel");
	}
}
