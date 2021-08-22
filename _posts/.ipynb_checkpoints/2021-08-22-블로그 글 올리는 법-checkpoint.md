---
layout: post
title: 블로그 글 작성 방법
categories: [잡담]
tags: [깃허브 블로그]
description: 글 올리는 순서
---

1. blog 정보가 있는 경로로 jupyter lab 을 실행한다

2. _posts 에서 쓰고싶은 글 작성한다
    * 이때 yyyy-mm-dd-{제목} 같은 양식을 지켜줘야한다.
3. blog 정보가 있는 경로로 bundle exec jekyll serve 실행

4. 글을 알맞게 수정한다

5. blog 정보가 있는 경로로 git push 한다

{% highlight ruby %}
    - git init
    - git add .
    - git commit -m "commit 내용"
    - git push origin main

{% endhighlight %}

<br>

#### 생각보다 글 올리는게 번거롭다.. 이렇게 하는게 맞는건가?

