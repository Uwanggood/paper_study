using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BamsongiController : MonoBehaviour
{
    public GameObject BamsongiGenerator;
    public void Shoot(Vector3 dir){
        GetComponent<AudioSource>().Play();
        GetComponent<Rigidbody>().AddForce(dir);
    }
    void OnCollisionEnter(Collision other){
       // if(other.gameObject.name=="Terrain")//Destroy(gameObject);
       if(other.gameObject.name=="Terrain"){
        //GetComponent<Rigidbody>().angularDrag = 30.0f;
        GetComponent<Rigidbody>().isKinematic = false;
       }
       else{
     //AudioSource.PlayClipAtPoint (Sound, transform.position);
        //AudioSource.clip = Resources.Load("Resource_Chapter8_get", typeof(AudioSource)) as AudioClip;
        //GetComponent<AudioSource>().Play();
       // GameObject delta = Resources.Load<AudioSource>("Resource_Chapter8_get", typeof(GameObject)) as GameObject;
         //delta.GetComponent<AudioSource>().Play();
         GetComponent<AudioSource>().Play();
         GetComponent<Rigidbody>().isKinematic = true;
        GetComponent<ParticleSystem>().Play();
        Destroy(gameObject,0.5f);
       }
        //GetComponent<Rigidbody>().isKinematic = true;
        //GetComponent<ParticleSystem>().Play();
    }
    // void Start(){
    //     // Shoot(new Vector3(0,200,2000));
    //     // https://bigballdiary.tistory.com/29?category=879176 �о��

    // }
}
// -10 ~ 20 까지

// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;

// public class BamsongiController : MonoBehaviour
// {
//     public GameObject BamsongiGenerator;
//     void Start(){
//          GetComponent<Rigidbody>().isKinematic = true;
//         Shoot(new Vector3(0,200,2000));
//         https://bigballdiary.tistory.com/29?category=879176 �о��

//     }
//     void Update(){
//          if (Input.GetMouseButtonDown(0)) // 마우스 클릭시 발동
//         {
//             GetComponent<Rigidbody>().isKinematic = false;
//             물체가 여러 물리력을 받도록 허용하는 코드
//             Shoot();
//             발사!!
//         }
//     }
//     public void Shoot(Vector3 dir){
//         GetComponent<Rigidbody>().AddForce(dir);
//     }
//     void OnCollisionEnter(Collision other){
//        // if(other.gameObject.name=="Terrain")//Destroy(gameObject);
//        if(other.gameObject.name=="Terrain"){
//         //GetComponent<Rigidbody>().angularDrag = 30.0f;
//         GetComponent<Rigidbody>().isKinematic = false;
//        }
//        else{
//          GetComponent<Rigidbody>().isKinematic = true;
//         GetComponent<ParticleSystem>().Play();
//        }
//         //GetComponent<Rigidbody>().isKinematic = true;
//         //GetComponent<ParticleSystem>().Play();
//     }
    
// }
// -10 ~ 20 까지
